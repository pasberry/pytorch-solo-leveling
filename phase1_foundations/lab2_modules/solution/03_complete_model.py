"""
Phase 1 Lab 2: Building a Complete Custom Model
Combine custom modules into a full model with hooks, state management, and best practices
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import json


class CustomBlock(nn.Module):
    """
    A residual block with custom components.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        activation: Activation function
        use_residual: Whether to use residual connection
        dropout: Dropout probability
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU(),
        use_residual: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_residual = use_residual and (in_features == out_features)

        self.fc1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.activation1 = activation
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.activation2 = activation
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First layer
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        # Second layer
        out = self.fc2(out)
        out = self.norm2(out)

        # Residual connection
        if self.use_residual:
            out = out + identity

        out = self.activation2(out)
        out = self.dropout2(out)

        return out


class CustomClassifier(nn.Module):
    """
    Complete custom classifier with advanced features:
    - Residual blocks
    - Multiple activation options
    - Forward hooks for feature extraction
    - State dict customization
    - Model statistics tracking

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        output_dim: Output dimension (number of classes)
        activation: Activation function name ('relu', 'gelu', 'silu')
        dropout: Dropout probability
        use_residual: Whether to use residual connections
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout = dropout
        self.use_residual = use_residual

        # Get activation function
        self.activation = self._get_activation(activation)

        # Build network
        self.blocks = nn.ModuleList()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Hidden blocks
        for i in range(len(hidden_dims)):
            out_dim = hidden_dims[i]
            block = CustomBlock(
                in_features=out_dim,
                out_features=out_dim,
                activation=self._get_activation(activation),
                use_residual=use_residual,
                dropout=dropout
            )
            self.blocks.append(block)

        # Output layer
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

        # Statistics tracking
        self.register_buffer('forward_count', torch.tensor(0))
        self._activations = {}
        self._hook_handles = []

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic forward counting."""
        self.forward_count += 1

        # Input projection
        x = self.input_proj(x)

        # Pass through blocks
        for i, block in enumerate(self.blocks):
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        return x

    def register_feature_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def get_activation(name):
            def hook(module, input, output):
                self._activations[name] = output.detach()
            return hook

        # Register hooks for each block
        for i, block in enumerate(self.blocks):
            handle = block.register_forward_hook(get_activation(f'block_{i}'))
            self._hook_handles.append(handle)

    def remove_feature_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._activations = {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations from last forward pass."""
        return self._activations

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        return {
            'architecture': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'num_blocks': len(self.blocks),
            'total_parameters': self.count_parameters(trainable_only=False),
            'trainable_parameters': self.count_parameters(trainable_only=True),
            'forward_count': self.forward_count.item()
        }

    def reset_statistics(self):
        """Reset internal statistics."""
        self.forward_count.zero_()

    def state_dict(self, *args, **kwargs):
        """Enhanced state dict with metadata."""
        state = super().state_dict(*args, **kwargs)

        # Add metadata
        state['_metadata'] = {
            'model_info': self.get_model_info(),
            'pytorch_version': torch.__version__
        }

        return state

    def load_state_dict(self, state_dict, strict=True):
        """Enhanced loading with metadata handling."""
        # Extract and remove metadata
        metadata = state_dict.pop('_metadata', None)

        if metadata:
            print(f"Loading model trained with PyTorch {metadata.get('pytorch_version', 'unknown')}")
            print(f"Model info: {metadata.get('model_info', {})}")

        # Load the actual state
        return super().load_state_dict(state_dict, strict=strict)

    def save_config(self, path: str):
        """Save model configuration to JSON."""
        config = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'dropout': self.dropout,
            'use_residual': self.use_residual
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_config(cls, path: str):
        """Load model from configuration file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)


def test_custom_classifier():
    """Test the custom classifier."""
    print("Testing CustomClassifier:")
    print("=" * 60)

    # Create model
    model = CustomClassifier(
        input_dim=100,
        hidden_dims=[256, 128, 64],
        output_dim=10,
        activation='gelu',
        dropout=0.1,
        use_residual=True
    )

    print(model)
    print()

    # Get model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 100)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward count: {model.forward_count.item()}")
    print()

    # Test with hooks
    model.register_feature_hooks()
    _ = model(x)
    activations = model.get_activations()

    print("Captured activations:")
    for name, act in activations.items():
        print(f"  {name}: {act.shape}")

    model.remove_feature_hooks()
    print()

    print("=" * 60)


def test_state_dict():
    """Test enhanced state dict functionality."""
    print("\nTesting Enhanced State Dict:")
    print("=" * 60)

    # Create and train a model briefly
    model1 = CustomClassifier(
        input_dim=50,
        hidden_dims=[128, 64],
        output_dim=5,
        activation='relu'
    )

    # Simulate training
    x = torch.randn(16, 50)
    for _ in range(5):
        _ = model1(x)

    print(f"Model 1 forward count: {model1.forward_count.item()}")

    # Save state
    state = model1.state_dict()
    print(f"State dict keys: {len(state.keys())}")
    print(f"Has metadata: {'_metadata' in state}")

    # Create new model and load state
    model2 = CustomClassifier(
        input_dim=50,
        hidden_dims=[128, 64],
        output_dim=5,
        activation='relu'
    )

    print("\nLoading state dict...")
    model2.load_state_dict(state)

    # Verify weights match
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert torch.allclose(param1, param2), f"Parameters {name1} don't match!"

    print("✓ Weights loaded successfully")
    print("=" * 60)


def test_training_eval_modes():
    """Test behavior in training vs evaluation mode."""
    print("\nTesting Training vs Evaluation Mode:")
    print("=" * 60)

    model = CustomClassifier(
        input_dim=50,
        hidden_dims=[128, 64],
        output_dim=5,
        dropout=0.5  # High dropout to see effect
    )

    x = torch.randn(100, 50)

    # Training mode
    model.train()
    train_outputs = []
    for _ in range(3):
        train_outputs.append(model(x))

    # Check that outputs differ (due to dropout)
    diff = (train_outputs[0] - train_outputs[1]).abs().mean().item()
    print(f"Difference between training runs: {diff:.6f}")
    print(f"Outputs vary due to dropout: {diff > 1e-6}")

    # Evaluation mode
    model.eval()
    eval_outputs = []
    for _ in range(3):
        eval_outputs.append(model(x))

    # Check that outputs are identical (no dropout)
    diff = (eval_outputs[0] - eval_outputs[1]).abs().mean().item()
    print(f"Difference between eval runs: {diff:.6f}")
    print(f"Outputs are deterministic: {diff < 1e-6}")

    print("=" * 60)


def benchmark_model_sizes():
    """Benchmark different model sizes."""
    print("\nBenchmarking Model Sizes:")
    print("=" * 60)

    configs = [
        {'name': 'Tiny', 'hidden_dims': [64, 32]},
        {'name': 'Small', 'hidden_dims': [128, 64, 32]},
        {'name': 'Medium', 'hidden_dims': [256, 128, 64]},
        {'name': 'Large', 'hidden_dims': [512, 256, 128, 64]},
    ]

    for config in configs:
        model = CustomClassifier(
            input_dim=100,
            hidden_dims=config['hidden_dims'],
            output_dim=10
        )

        params = model.count_parameters()
        print(f"{config['name']:8s}: {params:>10,} parameters")

    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)

    test_custom_classifier()
    test_state_dict()
    test_training_eval_modes()
    benchmark_model_sizes()

    print("\n✓ All tests passed!")
