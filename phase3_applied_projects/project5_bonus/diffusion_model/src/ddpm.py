"""
DDPM: Denoising Diffusion Probabilistic Models
Generate high-quality images by learning to denoise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (B,) - timestep indices

        Returns:
            embeddings: (B, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_dim)

        Returns:
            output: (B, C, H, W)
        """
        h = self.conv1(x)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)

        # Residual
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for denoising

    Architecture:
        Input (noisy image + time)
        ↓
        Encoder (downsample)
        ↓
        Bottleneck
        ↓
        Decoder (upsample with skip connections)
        ↓
        Output (predicted noise)
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=128, time_dim=256):
        super().__init__()

        self.time_dim = time_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2, base_channels * 2, time_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_dim)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4, base_channels * 4, time_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_dim)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4, time_dim),
            ResBlock(base_channels * 4, base_channels * 4, time_dim)
        )

        # Decoder (upsampling)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            ResBlock(base_channels * 4, base_channels * 2, time_dim),  # *4 due to skip connection
            ResBlock(base_channels * 2, base_channels * 2, time_dim)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            ResBlock(base_channels * 2, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim)
        )

        self.up1 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim)
        )

        # Final output
        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, time):
        """
        Args:
            x: (B, C, H, W) - noisy image
            time: (B,) - timestep

        Returns:
            noise_pred: (B, C, H, W) - predicted noise
        """
        # Time embedding
        time_emb = self.time_mlp(time)

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder with skip connections
        u3 = self.up3[0](b)  # Upsample
        u3 = torch.cat([u3, d2], dim=1)  # Skip connection
        for layer in self.up3[1:]:
            if isinstance(layer, ResBlock):
                u3 = layer(u3, time_emb)
            else:
                u3 = layer(u3)

        u2 = self.up2[0](u3)
        u2 = torch.cat([u2, d1], dim=1)
        for layer in self.up2[1:]:
            if isinstance(layer, ResBlock):
                u2 = layer(u2, time_emb)
            else:
                u2 = layer(u2)

        u1 = torch.cat([u2, x], dim=1)
        for layer in self.up1:
            if isinstance(layer, ResBlock):
                u1 = layer(u1, time_emb)
            else:
                u1 = layer(u1)

        # Output
        output = self.final(u1)

        return output


class DDPM:
    """
    Denoising Diffusion Probabilistic Model

    Forward process (add noise):
        x_0 → x_1 → x_2 → ... → x_T (pure noise)

    Reverse process (denoise):
        x_T → x_{T-1} → ... → x_1 → x_0 (generated image)
    """

    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Variance schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: Add noise to x_0 to get x_t

        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)

        Args:
            x_0: (B, C, H, W) - clean image
            t: (B,) - timestep
            noise: (B, C, H, W) - optional noise (if None, sample)

        Returns:
            x_t: (B, C, H, W) - noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t

    def p_losses(self, x_0, t, noise=None):
        """
        Compute training loss (simple MSE loss on noise prediction)

        Args:
            x_0: (B, C, H, W) - clean image
            t: (B,) - timestep

        Returns:
            loss: Scalar loss
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Add noise to get x_t
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse diffusion: Denoise x_t to get x_{t-1}

        Args:
            x_t: (B, C, H, W) - noisy image at timestep t
            t: (B,) - timestep

        Returns:
            x_{t-1}: (B, C, H, W) - denoised image
        """
        # Predict noise
        noise_pred = self.model(x_t, t)

        # Get coefficients
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Clip x_0
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # Get mean
        model_mean = (
            torch.sqrt(alpha_t) * (1 - self.alphas_cumprod_prev[t][:, None, None, None]) * x_t +
            torch.sqrt(self.alphas_cumprod_prev[t][:, None, None, None]) * beta_t * x_0_pred
        ) / (1 - alpha_cumprod_t)

        # Add noise (except for t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t][:, None, None, None]
            x_prev = model_mean + torch.sqrt(variance) * noise
        else:
            x_prev = model_mean

        return x_prev

    @torch.no_grad()
    def sample(self, shape, return_all_steps=False):
        """
        Generate samples by denoising from pure noise

        Args:
            shape: (B, C, H, W) - shape of images to generate
            return_all_steps: Whether to return all intermediate steps

        Returns:
            samples: (B, C, H, W) - generated images
            intermediates: List of intermediate steps (if return_all_steps=True)
        """
        batch_size = shape[0]
        device = self.device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = []

        # Denoise step by step
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

            if return_all_steps:
                intermediates.append(x.cpu())

        if return_all_steps:
            return x, intermediates
        return x


def train_ddpm():
    """Training loop for DDPM"""
    print("""
    # DDPM Training Loop

    model = UNet(in_channels=3, out_channels=3, base_channels=128)
    model = model.to(device)

    ddpm = DDPM(model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch.to(device)  # (B, 3, H, W), normalized to [-1, 1]

            # Sample random timesteps
            t = torch.randint(0, ddpm.timesteps, (images.size(0),), device=device)

            # Compute loss
            loss = ddpm.p_losses(images, t)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # Generate samples
        if epoch % 10 == 0:
            samples = ddpm.sample(shape=(16, 3, 32, 32))
            save_image(samples, f'samples_epoch_{epoch}.png')

    # Final generation
    samples, intermediates = ddpm.sample(shape=(64, 3, 32, 32), return_all_steps=True)
    print(f'Generated {samples.size(0)} samples')
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("DDPM: Denoising Diffusion Probabilistic Models")
    print("=" * 60)

    # Create model
    model = UNet(in_channels=3, out_channels=3, base_channels=128)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create DDPM
    ddpm = DDPM(model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device=device)

    print(f"\nModel created:")
    print(f"Architecture: U-Net")
    print(f"Timesteps: 1000")
    print(f"Device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Example forward diffusion
    print("\n" + "=" * 60)
    print("Example: Forward Diffusion (Adding Noise)")
    print("=" * 60)

    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 32, 32).to(device)
    x_0 = torch.clamp(x_0, -1, 1)  # Normalize to [-1, 1]

    timesteps_to_show = [0, 250, 500, 750, 999]

    print(f"Starting image: {x_0.shape}")
    for t_val in timesteps_to_show:
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        x_t = ddpm.q_sample(x_0, t)
        noise_level = x_t.std().item()
        print(f"  t={t_val:3d}: noise_level={noise_level:.4f}")

    # Example training loss
    print("\n" + "=" * 60)
    print("Example: Training Loss")
    print("=" * 60)

    t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
    loss = ddpm.p_losses(x_0, t)
    print(f"Training loss: {loss.item():.4f}")

    # Example sampling
    print("\n" + "=" * 60)
    print("Example: Sampling (Generation)")
    print("=" * 60)

    print("Generating 4 samples (this will take a moment)...")
    print("(In practice, use DDIM for faster sampling)")

    # For demo, use fewer timesteps
    ddpm_fast = DDPM(model, timesteps=50, beta_start=1e-4, beta_end=0.02, device=device)
    samples = ddpm_fast.sample(shape=(4, 3, 32, 32))

    print(f"Generated samples: {samples.shape}")
    print(f"Sample range: [{samples.min().item():.2f}, {samples.max().item():.2f}]")

    print("\n" + "=" * 60)
    print("DDPM Model Ready!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✓ Forward process: Gradually add noise (q)")
    print("✓ Reverse process: Learn to denoise (p)")
    print("✓ Training: Predict noise at random timesteps")
    print("✓ Sampling: Denoise from pure noise → image")
    print("✓ U-Net: Encoder-decoder with skip connections")
    print("✓ Time embedding: Condition on timestep")

    print("\n" + "=" * 60)
    print("Training Setup:")
    print("=" * 60)
    train_ddpm()

    print("\n" + "=" * 60)
    print("Tips for Better Performance:")
    print("=" * 60)
    print("1. Use cosine schedule instead of linear")
    print("2. Train on high-res images (256x256 or 512x512)")
    print("3. Use DDIM for faster sampling (50 steps vs 1000)")
    print("4. Add classifier-free guidance for conditional generation")
    print("5. Use latent diffusion (encode to latent space first)")
