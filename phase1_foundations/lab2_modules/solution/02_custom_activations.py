"""
Phase 1 Lab 2: Custom Activation Functions and Normalization
Learn to implement custom activation functions and normalization layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomReLU(nn.Module):
    """
    Custom implementation of ReLU activation.
    ReLU(x) = max(0, x)
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.clamp_(min=0)
        return x.clamp(min=0)

    def extra_repr(self) -> str:
        return f'inplace={self.inplace}'


class CustomGELU(nn.Module):
    """
    Custom implementation of GELU (Gaussian Error Linear Unit).
    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
    of the standard normal distribution.

    We use the approximation:
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    def __init__(self, approximate: str = 'tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == 'tanh':
            # Fast approximation
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        else:
            # Exact implementation
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CustomSiLU(nn.Module):
    """
    Custom implementation of SiLU (Sigmoid Linear Unit), also known as Swish.
    SiLU(x) = x * σ(x) where σ(x) is the sigmoid function
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class CustomLayerNorm(nn.Module):
    """
    Custom implementation of Layer Normalization.

    LayerNorm normalizes across the feature dimension:
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    Args:
        normalized_shape: Input shape (int or tuple)
        eps: Small value for numerical stability
        elementwise_affine: Whether to learn gamma and beta parameters
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learned affine transformation if enabled
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class CustomBatchNorm1d(nn.Module):
    """
    Custom implementation of 1D Batch Normalization.

    BatchNorm normalizes across the batch dimension:
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    During training, uses batch statistics.
    During evaluation, uses running statistics.

    Args:
        num_features: Number of features (channels)
        eps: Small value for numerical stability
        momentum: Momentum for running statistics
        affine: Whether to learn gamma and beta
        track_running_stats: Whether to track running statistics
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features) or (batch_size, num_features, length)

        if self.training:
            # Calculate batch statistics
            if x.dim() == 2:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            else:  # x.dim() == 3
                mean = x.mean(dim=[0, 2])
                var = x.var(dim=[0, 2], unbiased=False)

            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            var = self.running_var

        # Normalize
        if x.dim() == 2:
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        else:  # x.dim() == 3
            x_normalized = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)

        # Apply affine transformation
        if self.affine:
            if x.dim() == 2:
                x_normalized = x_normalized * self.weight + self.bias
            else:
                x_normalized = x_normalized * self.weight[None, :, None] + self.bias[None, :, None]

        return x_normalized


import math


def test_activations():
    """Test custom activation functions."""
    print("Testing Custom Activation Functions:")
    print("=" * 60)

    x = torch.randn(5, 10)
    print(f"Input shape: {x.shape}")

    # Test ReLU
    custom_relu = CustomReLU()
    torch_relu = nn.ReLU()
    assert torch.allclose(custom_relu(x), torch_relu(x)), "ReLU mismatch!"
    print("✓ CustomReLU matches nn.ReLU")

    # Test GELU
    custom_gelu = CustomGELU(approximate='tanh')
    torch_gelu = nn.GELU(approximate='tanh')
    assert torch.allclose(custom_gelu(x), torch_gelu(x), atol=1e-6), "GELU mismatch!"
    print("✓ CustomGELU matches nn.GELU")

    # Test SiLU
    custom_silu = CustomSiLU()
    torch_silu = nn.SiLU()
    assert torch.allclose(custom_silu(x), torch_silu(x)), "SiLU mismatch!"
    print("✓ CustomSiLU matches nn.SiLU")

    print("=" * 60)


def test_layer_norm():
    """Test custom LayerNorm."""
    print("\nTesting Custom LayerNorm:")
    print("=" * 60)

    x = torch.randn(32, 10)  # (batch_size, features)

    custom_ln = CustomLayerNorm(10)
    torch_ln = nn.LayerNorm(10)

    # Copy parameters to ensure same initialization
    with torch.no_grad():
        torch_ln.weight.copy_(custom_ln.weight)
        torch_ln.bias.copy_(custom_ln.bias)

    custom_output = custom_ln(x)
    torch_output = torch_ln(x)

    assert torch.allclose(custom_output, torch_output, atol=1e-6), "LayerNorm mismatch!"
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {custom_output.shape}")
    print("✓ CustomLayerNorm matches nn.LayerNorm")

    # Test gradient flow
    loss = custom_output.sum()
    loss.backward()
    print(f"✓ Gradients computed successfully")
    print(f"  Weight gradient shape: {custom_ln.weight.grad.shape}")

    print("=" * 60)


def test_batch_norm():
    """Test custom BatchNorm1d."""
    print("\nTesting Custom BatchNorm1d:")
    print("=" * 60)

    x = torch.randn(32, 10)  # (batch_size, features)

    custom_bn = CustomBatchNorm1d(10)
    torch_bn = nn.BatchNorm1d(10)

    # Copy parameters to ensure same initialization
    with torch.no_grad():
        torch_bn.weight.copy_(custom_bn.weight)
        torch_bn.bias.copy_(custom_bn.bias)

    # Test training mode
    custom_bn.train()
    torch_bn.train()

    custom_output = custom_bn(x)
    torch_output = torch_bn(x)

    assert torch.allclose(custom_output, torch_output, atol=1e-5), "BatchNorm training mismatch!"
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {custom_output.shape}")
    print("✓ CustomBatchNorm1d matches nn.BatchNorm1d (training mode)")

    # Test eval mode
    custom_bn.eval()
    torch_bn.eval()

    custom_eval = custom_bn(x)
    torch_eval = torch_bn(x)

    assert torch.allclose(custom_eval, torch_eval, atol=1e-5), "BatchNorm eval mismatch!"
    print("✓ CustomBatchNorm1d matches nn.BatchNorm1d (eval mode)")

    print("=" * 60)


def demo_normalization_effects():
    """Demonstrate the effect of normalization."""
    print("\nDemonstrating Normalization Effects:")
    print("=" * 60)

    # Create unnormalized data with different scales
    x = torch.randn(100, 5) * torch.tensor([1.0, 10.0, 100.0, 0.1, 0.01])

    print("Original data statistics:")
    print(f"  Mean: {x.mean(dim=0)}")
    print(f"  Std:  {x.std(dim=0)}")

    # Apply LayerNorm
    ln = CustomLayerNorm(5)
    x_ln = ln(x)
    print("\nAfter LayerNorm (per sample):")
    print(f"  Mean: {x_ln.mean(dim=0)}")
    print(f"  Std:  {x_ln.std(dim=0)}")

    # Apply BatchNorm
    bn = CustomBatchNorm1d(5)
    bn.train()
    x_bn = bn(x)
    print("\nAfter BatchNorm (across batch):")
    print(f"  Mean: {x_bn.mean(dim=0)}")
    print(f"  Std:  {x_bn.std(dim=0)}")

    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)

    test_activations()
    test_layer_norm()
    test_batch_norm()
    demo_normalization_effects()

    print("\n✓ All tests passed!")
