"""
Phase 1 Lab 2: Custom nn.Module - Linear Layer Implementation
Learn to build custom PyTorch modules from scratch
"""

import torch
import torch.nn as nn
import math


class CustomLinear(nn.Module):
    """
    Custom implementation of a linear (fully-connected) layer.
    This demonstrates the core concepts of building PyTorch modules.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Initialize bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters using proper initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = xW^T + b

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Matrix multiplication: x @ W^T
        output = torch.matmul(x, self.weight.t())

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class CustomMultiLayerPerceptron(nn.Module):
    """
    Custom MLP using our CustomLinear layers.
    Demonstrates composing custom modules.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function to use
        dropout: Dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1
    ):
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(CustomLinear(prev_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation or dropout)
        layers.append(CustomLinear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


def test_custom_linear():
    """Test our custom linear layer."""
    print("Testing CustomLinear layer:")
    print("=" * 60)

    # Create layer
    layer = CustomLinear(10, 5)
    print(f"Layer: {layer}")

    # Test forward pass
    x = torch.randn(32, 10)
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Compare with PyTorch's Linear
    torch_layer = nn.Linear(10, 5)
    torch_output = torch_layer(x)
    print(f"PyTorch Linear output shape: {torch_output.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"Weight gradient shape: {layer.weight.grad.shape}")
    print(f"Bias gradient shape: {layer.bias.grad.shape}")

    print("\n" + "=" * 60)


def test_mlp():
    """Test our custom MLP."""
    print("Testing CustomMultiLayerPerceptron:")
    print("=" * 60)

    # Create MLP
    mlp = CustomMultiLayerPerceptron(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        output_dim=10,
        dropout=0.1
    )

    print(mlp)
    print()

    # Count parameters
    total_params = sum(p.numel() for p in mlp.parameters())
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 20)
    output = mlp(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test training mode vs eval mode
    mlp.train()
    train_output = mlp(x)
    mlp.eval()
    eval_output = mlp(x)

    print(f"\nOutputs differ due to dropout: {not torch.allclose(train_output, eval_output)}")

    print("=" * 60)


def compare_performance():
    """Compare our CustomLinear with PyTorch's Linear."""
    print("\nPerformance Comparison:")
    print("=" * 60)

    import time

    # Create layers
    custom_layer = CustomLinear(1000, 500)
    torch_layer = nn.Linear(1000, 500)

    # Create input
    x = torch.randn(1000, 1000)

    # Time custom layer
    start = time.time()
    for _ in range(100):
        _ = custom_layer(x)
    custom_time = time.time() - start

    # Time PyTorch layer
    start = time.time()
    for _ in range(100):
        _ = torch_layer(x)
    torch_time = time.time() - start

    print(f"CustomLinear time: {custom_time:.4f}s")
    print(f"PyTorch Linear time: {torch_time:.4f}s")
    print(f"Relative performance: {custom_time / torch_time:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run tests
    test_custom_linear()
    print()
    test_mlp()
    compare_performance()

    print("\n✓ All tests passed!")
