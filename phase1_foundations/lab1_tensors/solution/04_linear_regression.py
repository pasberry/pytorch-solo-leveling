"""
Lab 1.4: Linear Regression from Scratch
Implement linear regression using only tensors and autograd (no nn.Module)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_data(n_samples=100, noise=0.1):
    """
    Generate synthetic data: y = 3x + 2 + noise

    Args:
        n_samples: Number of data points
        noise: Standard deviation of Gaussian noise

    Returns:
        X: Input features (n_samples, 1)
        y: Target values (n_samples, 1)
    """
    torch.manual_seed(42)
    X = torch.randn(n_samples, 1)
    true_w = 3.0
    true_b = 2.0
    y = true_w * X + true_b + noise * torch.randn(n_samples, 1)
    return X, y


def initialize_parameters():
    """
    Initialize parameters w and b randomly

    Returns:
        w: Weight parameter (requires grad)
        b: Bias parameter (requires grad)
    """
    torch.manual_seed(42)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


def forward(X, w, b):
    """
    Forward pass: compute predictions

    Args:
        X: Input features
        w: Weight
        b: Bias

    Returns:
        Predictions
    """
    return X @ w.T + b  # or X * w + b for 1D


def compute_loss(y_pred, y_true):
    """
    Compute Mean Squared Error loss

    Args:
        y_pred: Predictions
        y_true: Ground truth

    Returns:
        MSE loss (scalar)
    """
    return ((y_pred - y_true) ** 2).mean()


def train_step(X, y, w, b, learning_rate):
    """
    Single training step

    Args:
        X: Input features
        y: Target values
        w: Weight parameter
        b: Bias parameter
        learning_rate: Learning rate

    Returns:
        Loss value
    """
    # Forward pass
    y_pred = forward(X, w, b)

    # Compute loss
    loss = compute_loss(y_pred, y)

    # Backward pass
    loss.backward()

    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Zero gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()

    return loss.item()


def train(X, y, n_epochs=100, learning_rate=0.1):
    """
    Train linear regression model

    Args:
        X: Input features
        y: Target values
        n_epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        w, b: Trained parameters
        losses: Training loss history
    """
    # Initialize parameters
    w, b = initialize_parameters()

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        loss = train_step(X, y, w, b, learning_rate)
        losses.append(loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

    return w, b, losses


def visualize_results(X, y, w, b, losses):
    """
    Visualize training results

    Args:
        X: Input features
        y: Target values
        w: Trained weight
        b: Trained bias
        losses: Training loss history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Data and fitted line
    X_np = X.detach().numpy()
    y_np = y.detach().numpy()

    axes[0].scatter(X_np, y_np, alpha=0.5, label='Data')

    # Plot fitted line
    X_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = forward(X_line, w, b)
    axes[0].plot(X_line.detach().numpy(), y_line.detach().numpy(),
                 'r-', linewidth=2, label='Fitted line')

    # Plot true line
    true_w, true_b = 3.0, 2.0
    y_true_line = true_w * X_line + true_b
    axes[0].plot(X_line.numpy(), y_true_line.numpy(),
                 'g--', linewidth=2, label='True line')

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('Linear Regression: Data and Fitted Line')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Training loss
    axes[1].plot(losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Training Loss Over Time')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'linear_regression_results.png'")
    plt.close()


def main():
    print("=" * 60)
    print("Lab 1.4: Linear Regression from Scratch")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data: y = 3x + 2 + noise")
    X, y = generate_data(n_samples=100, noise=0.5)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Train model
    print("\n2. Training linear regression model")
    print("-" * 60)
    w, b, losses = train(X, y, n_epochs=100, learning_rate=0.1)

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"True parameters:    w = 3.000, b = 2.000")
    print(f"Learned parameters: w = {w.item():.3f}, b = {b.item():.3f}")
    print(f"Final loss: {losses[-1]:.4f}")

    # Visualize
    print("\n3. Visualizing results")
    visualize_results(X, y, w, b, losses)

    # Bonus: Make predictions
    print("\n4. Making predictions")
    print("-" * 60)
    X_test = torch.tensor([[0.0], [1.0], [2.0], [-1.0]])
    y_pred = forward(X_test, w, b)

    print("Test predictions:")
    for x, y in zip(X_test, y_pred):
        y_true = 3.0 * x.item() + 2.0
        print(f"  x={x.item():5.1f} -> pred={y.item():6.3f}, true={y_true:6.3f}")

    print("\n" + "=" * 60)
    print("Lab 1.4 Complete!")
    print("=" * 60)
    print("\nKey Concepts Covered:")
    print("1. Data generation and preparation")
    print("2. Parameter initialization")
    print("3. Forward pass (prediction)")
    print("4. Loss computation (MSE)")
    print("5. Backward pass (autograd)")
    print("6. Parameter updates (gradient descent)")
    print("7. Training loop")
    print("8. Visualization")
    print("\nYou've just trained your first model with PyTorch!")


if __name__ == "__main__":
    main()
