"""
Phase 3 Project 2: Video Understanding
Implement video classification with 3D CNNs and temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Conv3DBlock(nn.Module):
    """3D Convolutional block for video."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class C3DModel(nn.Module):
    """
    3D CNN for video classification (C3D-style).

    Input: (batch, channels, time, height, width)
    """
    def __init__(self, num_classes: int = 400, dropout: float = 0.5):
        super().__init__()

        # Conv layers
        self.conv1 = Conv3DBlock(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = Conv3DBlock(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv3DBlock(128, 256)
        self.conv3b = Conv3DBlock(256, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv3DBlock(256, 512)
        self.conv4b = Conv3DBlock(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = Conv3DBlock(512, 512)
        self.conv5b = Conv3DBlock(512, 512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # FC layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 3, frames, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.pool5(self.conv5b(self.conv5a(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SlowFastBlock(nn.Module):
    """Block for SlowFast network with lateral connections."""
    def __init__(
        self,
        slow_channels: int,
        fast_channels: int,
        out_channels: int,
        alpha: int = 8
    ):
        super().__init__()
        self.alpha = alpha

        # Slow pathway
        self.slow_conv = Conv3DBlock(slow_channels, out_channels)

        # Fast pathway
        self.fast_conv = Conv3DBlock(fast_channels, out_channels // alpha)

        # Lateral connection (fast -> slow)
        self.lateral = nn.Conv3d(
            out_channels // alpha,
            out_channels,
            kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1),
            padding=(2, 0, 0),
            bias=False
        )

    def forward(self, x_slow, x_fast):
        """
        Args:
            x_slow: Slow pathway features
            x_fast: Fast pathway features
        Returns:
            Updated slow and fast features
        """
        # Process both pathways
        slow_out = self.slow_conv(x_slow)
        fast_out = self.fast_conv(x_fast)

        # Lateral connection: fuse fast into slow
        lateral_out = self.lateral(fast_out)
        slow_out = slow_out + lateral_out

        return slow_out, fast_out


class SlowFastModel(nn.Module):
    """
    SlowFast network for video understanding.
    - Slow pathway: High spatial resolution, low temporal rate
    - Fast pathway: Low spatial resolution, high temporal rate
    """
    def __init__(self, num_classes: int = 400, alpha: int = 8):
        super().__init__()
        self.alpha = alpha

        # Input conv
        self.slow_conv1 = Conv3DBlock(3, 64, stride=1)
        self.fast_conv1 = Conv3DBlock(3, 8, stride=1)  # Fewer channels

        # Blocks
        self.block1 = SlowFastBlock(64, 8, 128, alpha)
        self.block2 = SlowFastBlock(128, 16, 256, alpha)
        self.block3 = SlowFastBlock(256, 32, 512, alpha)

        # Pooling
        self.slow_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fast_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier
        self.fc = nn.Linear(512 + 64, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, frames, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        # Split into slow and fast pathways
        # Slow: sample every alpha frames
        x_slow = x[:, :, ::self.alpha, :, :]
        # Fast: all frames, lower resolution
        x_fast = F.interpolate(x, scale_factor=(1, 0.5, 0.5), mode='trilinear')

        # Initial conv
        x_slow = self.slow_conv1(x_slow)
        x_fast = self.fast_conv1(x_fast)

        # Blocks with lateral connections
        x_slow, x_fast = self.block1(x_slow, x_fast)
        x_slow, x_fast = self.block2(x_slow, x_fast)
        x_slow, x_fast = self.block3(x_slow, x_fast)

        # Pool both pathways
        x_slow = self.slow_pool(x_slow).flatten(1)
        x_fast = self.fast_pool(x_fast).flatten(1)

        # Concatenate and classify
        x = torch.cat([x_slow, x_fast], dim=1)
        x = self.fc(x)

        return x


class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module (TSM) for efficient video modeling.
    Shifts feature channels along temporal dimension.
    """
    def __init__(self, num_channels: int, num_frames: int, shift_div: int = 8):
        super().__init__()
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.shift_div = shift_div

    def forward(self, x):
        """
        Args:
            x: (batch, channels, frames, H, W)
        """
        batch, channels, frames, h, w = x.shape

        # Reshape to separate temporal dimension
        x = x.view(batch, channels, frames, h * w)

        # Split channels into 3 groups
        fold = channels // self.shift_div
        out = torch.zeros_like(x)

        # Shift left (past)
        out[:, :fold, 1:, :] = x[:, :fold, :-1, :]
        out[:, :fold, 0, :] = x[:, :fold, 0, :]

        # Shift right (future)
        out[:, fold:2*fold, :-1, :] = x[:, fold:2*fold, 1:, :]
        out[:, fold:2*fold, -1, :] = x[:, fold:2*fold, -1, :]

        # No shift (present)
        out[:, 2*fold:, :, :] = x[:, 2*fold:, :, :]

        return out.view(batch, channels, frames, h, w)


def demo_c3d():
    """Demonstrate C3D model."""
    print("=" * 70)
    print("DEMO: C3D Video Classification")
    print("=" * 70)

    # Model
    model = C3DModel(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Input: 16 frames of 112x112 RGB video
    x = torch.randn(2, 3, 16, 112, 112)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Show feature map evolution
    print("\nFeature progression:")
    model.eval()
    with torch.no_grad():
        print(f"  Input: {x.shape}")
        x1 = model.pool1(model.conv1(x))
        print(f"  After block 1: {x1.shape}")
        x2 = model.pool2(model.conv2(x1))
        print(f"  After block 2: {x2.shape}")

    print("=" * 70)


def demo_slowfast():
    """Demonstrate SlowFast model."""
    print("\n" + "=" * 70)
    print("DEMO: SlowFast Video Understanding")
    print("=" * 70)

    # Model
    model = SlowFastModel(num_classes=10, alpha=8)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Input: 64 frames
    x = torch.randn(2, 3, 64, 224, 224)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    print("\nSlowFast architecture:")
    print("  Slow pathway: Every 8th frame, high resolution")
    print("  Fast pathway: All frames, lower resolution")
    print("  Lateral connections fuse fast features into slow")

    print("=" * 70)


def demo_temporal_shift():
    """Demonstrate Temporal Shift Module."""
    print("\n" + "=" * 70)
    print("DEMO: Temporal Shift Module")
    print("=" * 70)

    # Create TSM
    tsm = TemporalShiftModule(num_channels=64, num_frames=8, shift_div=8)

    # Input features
    x = torch.randn(2, 64, 8, 14, 14)
    print(f"Input shape: {x.shape}")

    # Apply TSM
    shifted = tsm(x)
    print(f"Shifted shape: {shifted.shape}")

    print("\nTSM shifts 1/8 of channels forward and 1/8 backward in time")
    print("This enables temporal modeling without 3D convolutions")

    print("=" * 70)


def train_video_classifier():
    """Train video classifier."""
    print("\n" + "=" * 70)
    print("Training Video Classifier")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model (using smaller C3D for speed)
    model = C3DModel(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data
    num_epochs = 3
    batch_size = 4
    num_frames = 16

    print(f"Device: {device}")
    print(f"Training for {num_epochs} epochs\n")

    for epoch in range(num_epochs):
        model.train()

        # Generate batch
        videos = torch.randn(batch_size, 3, num_frames, 112, 112).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)

        # Forward
        outputs = model(videos)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).sum().item() / batch_size * 100

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_c3d()
    demo_slowfast()
    demo_temporal_shift()
    train_video_classifier()

    print("\n✓ Video understanding demonstrations complete!")
