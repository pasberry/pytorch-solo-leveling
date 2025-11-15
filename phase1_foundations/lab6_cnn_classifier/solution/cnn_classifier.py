"""
Phase 1 Lab 6: CNN Image Classifier
Build a complete CNN classifier with modern architectures and techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """Small ResNet-style architecture."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DenseBlock(nn.Module):
    """Dense block for DenseNet-style architecture."""
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (efficient mobile architecture)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class EfficientCNN(nn.Module):
    """Efficient CNN using depthwise separable convolutions."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def demo_architectures():
    """Compare different CNN architectures."""
    print("=" * 70)
    print("DEMO: CNN Architectures Comparison")
    print("=" * 70)

    input_shape = (1, 3, 32, 32)  # Batch=1, Channels=3, Height=32, Width=32
    x = torch.randn(input_shape)

    models = {
        'SimpleCNN': SimpleCNN(),
        'ResNet': ResNet(),
        'EfficientCNN': EfficientCNN()
    }

    print(f"Input shape: {input_shape}\n")

    for name, model in models.items():
        output = model(x)
        params = count_parameters(model)

        print(f"{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")
        print()

    print("=" * 70)


def train_and_evaluate():
    """Train and evaluate CNN classifier."""
    print("\n" + "=" * 70)
    print("DEMO: Training CNN Classifier")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Create synthetic dataset (simulating 32x32 RGB images)
    train_images = torch.randn(1000, 3, 32, 32)
    train_labels = torch.randint(0, 10, (1000,))
    val_images = torch.randn(200, 3, 32, 32)
    val_labels = torch.randint(0, 10, (200,))

    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images, val_labels), batch_size=32, shuffle=False)

    # Create model
    model = ResNet(in_channels=3, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model: ResNet")
    print(f"Parameters: {count_parameters(model):,}\n")

    # Train for a few epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Print metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_acc:.2f}%")

    print("\n" + "=" * 70)


def demo_feature_visualization():
    """Visualize CNN features."""
    print("\n" + "=" * 70)
    print("DEMO: Feature Visualization")
    print("=" * 70)

    model = SimpleCNN()
    x = torch.randn(1, 3, 32, 32)

    # Forward through each layer and print shapes
    print("Feature map shapes through network:\n")

    features = model.features
    current = x
    print(f"Input: {current.shape}")

    for i, layer in enumerate(features):
        current = layer(current)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"After {layer.__class__.__name__} {i}: {current.shape}")

    print("\n" + "=" * 70)


def demo_receptive_field():
    """Calculate receptive field."""
    print("\n" + "=" * 70)
    print("DEMO: Receptive Field Calculation")
    print("=" * 70)

    print("Receptive field grows with each layer:\n")

    layers = [
        ("Conv 3x3", 3, 1),
        ("Conv 3x3", 3, 1),
        ("MaxPool 2x2", 2, 2),
        ("Conv 3x3", 3, 1),
        ("Conv 3x3", 3, 1),
        ("MaxPool 2x2", 2, 2),
    ]

    receptive_field = 1
    stride_product = 1

    print(f"Initial: RF = {receptive_field}")

    for name, kernel, stride in layers:
        receptive_field = receptive_field + (kernel - 1) * stride_product
        stride_product *= stride
        print(f"After {name}: RF = {receptive_field}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_architectures()
    train_and_evaluate()
    demo_feature_visualization()
    demo_receptive_field()

    print("\n✓ All CNN demonstrations complete!")
