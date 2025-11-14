"""
Lab 1.2: Tensor Operations
Learn essential tensor operations and mathematical functions
"""

import torch
import numpy as np


def main():
    print("=" * 60)
    print("Lab 1.2: Tensor Operations")
    print("=" * 60)

    # 1. Element-wise operations
    print("\n1. Element-wise Operations")
    print("-" * 60)

    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([5, 6, 7, 8])

    # Addition
    c = a + b
    print(f"a + b = {c}")
    print(f"torch.add(a, b) = {torch.add(a, b)}")

    # Subtraction
    print(f"b - a = {b - a}")

    # Multiplication (element-wise)
    print(f"a * b = {a * b}")

    # Division
    print(f"b / a = {b / a}")

    # Power
    print(f"a ** 2 = {a ** 2}")

    # In-place operations (end with _)
    a_copy = a.clone()
    a_copy.add_(5)  # In-place addition
    print(f"After a.add_(5): {a_copy}")

    # 2. Matrix operations
    print("\n2. Matrix Operations")
    print("-" * 60)

    A = torch.randn(3, 4)
    B = torch.randn(4, 5)

    # Matrix multiplication
    C = torch.matmul(A, B)  # or A @ B
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A @ B shape: {C.shape}")
    print(f"A @ B:\n{C}")

    # Transpose
    A_T = A.T  # or A.transpose(0, 1)
    print(f"\nA.T shape: {A_T.shape}")

    # Batch matrix multiplication
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    batch_result = torch.bmm(batch1, batch2)
    print(f"\nBatch matmul: ({10}, {3}, {4}) @ ({10}, {4}, {5}) = {batch_result.shape}")

    # 3. Broadcasting
    print("\n3. Broadcasting")
    print("-" * 60)

    # Broadcasting allows operations on tensors of different shapes
    x = torch.randn(3, 4)
    y = torch.randn(4)  # Will be broadcast to (3, 4)

    z = x + y
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"x + y shape: {z.shape}")
    print("Broadcasting y from (4,) to (3, 4)")

    # More complex broadcasting
    a = torch.randn(2, 1, 4)
    b = torch.randn(3, 1)
    c = a + b
    print(f"\na: {a.shape}, b: {b.shape}")
    print(f"a + b shape: {c.shape} (broadcast to (2, 3, 4))")

    # 4. Reduction operations
    print("\n4. Reduction Operations")
    print("-" * 60)

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"Original tensor:\n{x}")

    # Sum
    print(f"\nSum (all): {x.sum()}")
    print(f"Sum (dim=0): {x.sum(dim=0)}")  # Sum over rows
    print(f"Sum (dim=1): {x.sum(dim=1)}")  # Sum over columns
    print(f"Sum (keepdim=True):\n{x.sum(dim=1, keepdim=True)}")

    # Mean
    print(f"\nMean: {x.mean()}")
    print(f"Mean (dim=0): {x.mean(dim=0)}")

    # Max and min
    print(f"\nMax: {x.max()}")
    print(f"Max (dim=1): {x.max(dim=1)}")  # Returns values and indices
    print(f"Max values (dim=1): {x.max(dim=1).values}")
    print(f"Max indices (dim=1): {x.max(dim=1).indices}")

    # ArgMax
    print(f"\nArgmax: {x.argmax()}")  # Index of max element
    print(f"Argmax (dim=1): {x.argmax(dim=1)}")

    # 5. Statistical operations
    print("\n5. Statistical Operations")
    print("-" * 60)

    x = torch.randn(1000)

    print(f"Mean: {x.mean():.4f}")
    print(f"Std: {x.std():.4f}")
    print(f"Var: {x.var():.4f}")
    print(f"Min: {x.min():.4f}")
    print(f"Max: {x.max():.4f}")

    # 6. Comparison operations
    print("\n6. Comparison Operations")
    print("-" * 60)

    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([5, 4, 3, 2, 1])

    print(f"a: {a}")
    print(f"b: {b}")
    print(f"a > b: {a > b}")
    print(f"a == b: {a == b}")
    print(f"torch.eq(a, b): {torch.eq(a, b)}")

    # 7. Useful functions
    print("\n7. Useful Functions")
    print("-" * 60)

    x = torch.randn(3, 4)

    # Clamp (clip values)
    clamped = torch.clamp(x, min=-0.5, max=0.5)
    print(f"Original:\n{x}")
    print(f"Clamped to [-0.5, 0.5]:\n{clamped}")

    # Absolute value
    print(f"\nAbs:\n{torch.abs(x)}")

    # Round
    print(f"\nRound:\n{torch.round(x)}")

    # Concatenation
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    concat_0 = torch.cat([a, b], dim=0)  # Stack vertically
    concat_1 = torch.cat([a, b], dim=1)  # Stack horizontally
    print(f"\na shape: {a.shape}, b shape: {b.shape}")
    print(f"cat(dim=0) shape: {concat_0.shape}")
    print(f"cat(dim=1) shape: {concat_1.shape}")

    # Stack (adds new dimension)
    stacked = torch.stack([a, b], dim=0)
    print(f"stack(dim=0) shape: {stacked.shape}")

    # Split
    x = torch.arange(12).reshape(3, 4)
    splits = torch.split(x, 2, dim=0)  # Split into chunks of size 2
    print(f"\nOriginal: {x.shape}")
    print(f"After split(2, dim=0): {[s.shape for s in splits]}")

    # 8. Advanced indexing
    print("\n8. Advanced Indexing")
    print("-" * 60)

    x = torch.arange(12).reshape(3, 4)
    print(f"Original:\n{x}")

    # Gather
    indices = torch.tensor([[0, 1], [2, 3]])
    gathered = torch.gather(x, 1, indices)
    print(f"\nGather with indices {indices.tolist()}:\n{gathered}")

    # Where (conditional selection)
    condition = x > 5
    result = torch.where(condition, x, torch.zeros_like(x))
    print(f"\nWhere x > 5:\n{result}")

    # 9. Performance tips
    print("\n9. Performance Tips")
    print("-" * 60)

    # In-place operations save memory
    x = torch.randn(1000, 1000)

    # Not in-place (creates new tensor)
    y = x + 1

    # In-place (modifies x)
    x.add_(1)

    print("Use in-place operations (add_, mul_, etc.) when possible")
    print("Avoid python loops - use vectorized operations")
    print("Use torch.no_grad() during inference")

    print("\n" + "=" * 60)
    print("Lab 1.2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
