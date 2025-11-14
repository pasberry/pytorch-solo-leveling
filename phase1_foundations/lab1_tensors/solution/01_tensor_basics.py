"""
Lab 1.1: Tensor Basics
Learn how to create and manipulate PyTorch tensors
"""

import torch
import numpy as np


def main():
    print("=" * 60)
    print("Lab 1.1: Tensor Basics")
    print("=" * 60)

    # 1. Creating tensors
    print("\n1. Creating Tensors")
    print("-" * 60)

    # Zeros tensor
    zeros = torch.zeros(3, 3)
    print(f"3x3 zeros tensor:\n{zeros}")
    print(f"Shape: {zeros.shape}, Dtype: {zeros.dtype}")

    # Random normal tensor
    randn = torch.randn(2, 4)
    print(f"\n2x4 random normal tensor:\n{randn}")

    # From Python list
    list_data = [[1, 2, 3], [4, 5, 6]]
    from_list = torch.tensor(list_data)
    print(f"\nTensor from list:\n{from_list}")

    # From NumPy array
    np_array = np.array([[1, 2], [3, 4]])
    from_numpy = torch.from_numpy(np_array)
    print(f"\nTensor from NumPy:\n{from_numpy}")

    # Range of values
    range_tensor = torch.arange(0, 10)
    print(f"\nRange tensor (0-9): {range_tensor}")

    # Linspace
    linspace_tensor = torch.linspace(0, 1, steps=5)
    print(f"Linspace (0-1, 5 steps): {linspace_tensor}")

    # 2. Tensor attributes
    print("\n2. Tensor Attributes")
    print("-" * 60)

    x = torch.randn(2, 3, 4)
    print(f"Shape: {x.shape}")  # Same as x.size()
    print(f"Dtype: {x.dtype}")
    print(f"Device: {x.device}")
    print(f"Requires grad: {x.requires_grad}")
    print(f"Number of elements: {x.numel()}")
    print(f"Number of dimensions: {x.ndim}")

    # 3. Reshaping tensors
    print("\n3. Reshaping Tensors")
    print("-" * 60)

    # Create 1D tensor
    x = torch.arange(12)
    print(f"Original shape: {x.shape}")

    # Reshape to 2D
    x_2d = x.reshape(3, 4)
    print(f"Reshaped to (3, 4):\n{x_2d}")

    # Reshape to 3D
    x_3d = x.reshape(2, 2, 3)
    print(f"Reshaped to (2, 2, 3):\n{x_3d}")

    # Use -1 for automatic dimension inference
    x_auto = x.reshape(2, -1)  # Will be (2, 6)
    print(f"Reshaped with -1 to (2, -1): shape={x_auto.shape}")

    # View (similar to reshape but with memory constraints)
    x_view = x.view(4, 3)
    print(f"Using view(): {x_view.shape}")

    # Squeeze and unsqueeze
    x = torch.randn(1, 3, 1, 4)
    print(f"\nOriginal shape with singleton dims: {x.shape}")
    x_squeezed = x.squeeze()  # Remove all singleton dimensions
    print(f"After squeeze(): {x_squeezed.shape}")
    x_unsqueezed = x_squeezed.unsqueeze(0)  # Add dimension at position 0
    print(f"After unsqueeze(0): {x_unsqueezed.shape}")

    # 4. Indexing and slicing
    print("\n4. Indexing and Slicing")
    print("-" * 60)

    x = torch.arange(20).reshape(4, 5)
    print(f"Original tensor:\n{x}")

    # Basic indexing
    print(f"\nx[0]: {x[0]}")  # First row
    print(f"x[:, 0]: {x[:, 0]}")  # First column
    print(f"x[1:3]: \n{x[1:3]}")  # Rows 1 and 2
    print(f"x[1:3, 2:4]: \n{x[1:3, 2:4]}")  # Submatrix

    # Advanced indexing
    indices = torch.tensor([0, 2, 3])
    print(f"\nx[indices]: \n{x[indices]}")  # Select specific rows

    # Boolean indexing
    mask = x > 10
    print(f"\nMask (x > 10):\n{mask}")
    print(f"x[mask]: {x[mask]}")  # Elements > 10

    # 5. Tensor dtypes
    print("\n5. Data Types")
    print("-" * 60)

    # Different dtypes
    float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)

    print(f"Float32: {float_tensor}, dtype={float_tensor.dtype}")
    print(f"Int64: {int_tensor}, dtype={int_tensor.dtype}")
    print(f"Bool: {bool_tensor}, dtype={bool_tensor.dtype}")

    # Type conversion
    x = torch.randn(3, 3)
    x_int = x.int()
    x_double = x.double()
    print(f"\nOriginal dtype: {x.dtype}")
    print(f"After .int(): {x_int.dtype}")
    print(f"After .double(): {x_double.dtype}")

    # 6. Moving between CPU and GPU
    print("\n6. Device Management")
    print("-" * 60)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create tensor on specific device
    x_cpu = torch.randn(3, 3)
    print(f"x_cpu device: {x_cpu.device}")

    # Move to GPU (if available)
    if torch.cuda.is_available():
        x_gpu = x_cpu.to(device)
        print(f"x_gpu device: {x_gpu.device}")

        # Or create directly on GPU
        x_gpu2 = torch.randn(3, 3, device=device)
        print(f"x_gpu2 device: {x_gpu2.device}")

        # Move back to CPU
        x_back = x_gpu.cpu()
        print(f"x_back device: {x_back.device}")
    else:
        print("CUDA not available, skipping GPU examples")

    print("\n" + "=" * 60)
    print("Lab 1.1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
