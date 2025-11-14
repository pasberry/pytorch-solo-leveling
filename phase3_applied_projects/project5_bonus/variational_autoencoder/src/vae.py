"""
Variational Autoencoder (VAE)
Probabilistic generative model with latent variables
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder: Maps input x to latent distribution parameters (μ, σ)

    q(z|x) = N(z; μ(x), σ²(x)I)
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Output layers for μ and log(σ²)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim)

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: Maps latent z to reconstruction x'

    p(x|z) = Bernoulli(x; π(z))  [for binary data]
    or
    p(x|z) = N(x; μ(z), σ²I)  [for continuous data]
    """

    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, output_dim)
        """
        h = self.decoder(z)
        x_recon = self.fc_out(h)
        return x_recon


class VAE(nn.Module):
    """
    Variational Autoencoder

    Loss = Reconstruction Loss + KL Divergence

    - Reconstruction: How well can we reconstruct input
    - KL Divergence: Regularize latent space to be close to N(0, I)
    """

    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        This allows backprop through sampling

        Args:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)

        Returns:
            z: (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Args:
            x: (B, input_dim)

        Returns:
            x_recon: (B, input_dim) - reconstruction
            mu: (B, latent_dim) - mean of q(z|x)
            logvar: (B, latent_dim) - log variance of q(z|x)
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def sample(self, num_samples, device='cpu'):
        """
        Generate samples by sampling z ~ N(0, I) and decoding

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            samples: (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    def encode(self, x):
        """Get latent representation"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence

    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Mean of q(z|x)
        logvar: Log variance of q(z|x)
        beta: Weight for KL term (β-VAE for disentanglement)

    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (binary cross-entropy for [0,1] images)
    recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')

    # For continuous data, use MSE:
    # recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss


class BetaVAE(VAE):
    """
    β-VAE: VAE with adjustable β for disentanglement

    Higher β → more disentangled latent factors
    """

    def __init__(self, input_dim=784, hidden_dims=[512, 256], latent_dim=20, beta=4.0):
        super().__init__(input_dim, hidden_dims, latent_dim)
        self.beta = beta


class ConvVAE(nn.Module):
    """
    Convolutional VAE for images
    """

    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 4x4 -> 2x2
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        h = self.fc_decode(z)
        h = h.view(-1, 256, 2, 2)
        x_recon = self.decoder(h)

        return x_recon, mu, logvar


if __name__ == "__main__":
    print("=" * 60)
    print("Variational Autoencoder (VAE)")
    print("=" * 60)

    # Create VAE
    input_dim = 784  # 28x28 MNIST images
    latent_dim = 20
    vae = VAE(input_dim=input_dim, hidden_dims=[512, 256], latent_dim=latent_dim)

    print(f"\nVAE Model:")
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Latent dimension: {latent_dim}")

    # Example forward pass
    batch_size = 32
    x = torch.rand(batch_size, input_dim)  # Random images

    x_recon, mu, logvar = vae(x)

    print(f"\nExample:")
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"Latent μ: {mu.shape}")
    print(f"Latent logvar: {logvar.shape}")

    # Compute loss
    loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

    print(f"\nLoss:")
    print(f"Total: {loss.item():.2f}")
    print(f"Reconstruction: {recon_loss.item():.2f}")
    print(f"KL Divergence: {kl_loss.item():.2f}")

    # Sample from prior
    samples = vae.sample(num_samples=10)
    print(f"\nGenerated samples: {samples.shape}")

    # Latent space interpolation
    print("\n" + "=" * 60)
    print("Latent Space Interpolation")
    print("=" * 60)

    z1 = vae.encode(x[0:1])
    z2 = vae.encode(x[1:2])

    print(f"z1: {z1.shape}")
    print(f"z2: {z2.shape}")

    # Interpolate
    alphas = torch.linspace(0, 1, 5)
    interpolations = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        x_interp = vae.decoder(z_interp)
        interpolations.append(x_interp)

    print(f"Interpolated {len(interpolations)} images")

    print("\n" + "=" * 60)
    print("VAE Complete!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✓ Encoder: x → (μ, σ) - parameters of q(z|x)")
    print("✓ Reparameterization: z = μ + σε, ε ~ N(0,I)")
    print("✓ Decoder: z → x' - reconstruct from latent")
    print("✓ Loss: Reconstruction + KL divergence")
    print("✓ KL: Regularize latent space to N(0, I)")
    print("✓ Sampling: z ~ N(0, I) → decoder → generated image")
    print("✓ β-VAE: Larger β → more disentangled latents")

    print("\n" + "=" * 60)
    print("Training Skeleton:")
    print("=" * 60)
    print("""
    vae = VAE(input_dim=784, latent_dim=20)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch.flatten(1)  # Flatten images

            # Forward
            x_recon, mu, logvar = vae(x)

            # Compute loss
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=1.0)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.2f}')

        # Generate samples
        with torch.no_grad():
            samples = vae.sample(64)
            save_image(samples.view(-1, 1, 28, 28), f'samples_{epoch}.png')
    """)
