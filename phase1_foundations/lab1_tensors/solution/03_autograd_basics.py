"""
Lab 1.3: Autograd Basics
Learn PyTorch's automatic differentiation
"""

import torch


def main():
    print("=" * 60)
    print("Lab 1.3: Autograd Basics")
    print("=" * 60)

    # 1. Basic gradient computation
    print("\n1. Basic Gradient Computation")
    print("-" * 60)

    # Create tensor with requires_grad=True to track computations
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x = {x}")
    print(f"x.requires_grad = {x.requires_grad}")

    # Perform operations
    y = x ** 2 + 3 * x + 1
    print(f"y = x^2 + 3x + 1 = {y}")

    # Compute gradients
    y.backward()  # dy/dx = 2x + 3
    print(f"dy/dx at x=2: {x.grad}")  # Should be 2*2 + 3 = 7

    # 2. Multi-variable gradients
    print("\n2. Multi-variable Gradients")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)

    z = x ** 2 + y ** 3 + x * y
    print(f"z = x^2 + y^3 + xy = {z}")

    z.backward()

    print(f"dz/dx = 2x + y = {x.grad}")  # 2*2 + 3 = 7
    print(f"dz/dy = 3y^2 + x = {y.grad}")  # 3*3^2 + 2 = 29

    # 3. Gradient accumulation
    print("\n3. Gradient Accumulation")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)

    # First computation
    y1 = x ** 2
    y1.backward()
    print(f"After first backward: x.grad = {x.grad}")

    # Second computation (gradients accumulate!)
    y2 = x ** 3
    y2.backward()
    print(f"After second backward: x.grad = {x.grad} (accumulated!)")

    # Need to zero gradients manually
    x.grad.zero_()
    y3 = x ** 2
    y3.backward()
    print(f"After zeroing and new backward: x.grad = {x.grad}")

    # 4. Vector-Jacobian product
    print("\n4. Vector-Jacobian Product")
    print("-" * 60)

    # When output is not a scalar, need to provide gradient argument
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2
    print(f"x = {x}")
    print(f"y = x^2 = {y}")

    # Provide gradient weights for each output
    gradient = torch.tensor([1.0, 1.0, 1.0])
    y.backward(gradient=gradient)
    print(f"dy/dx = 2x = {x.grad}")

    # 5. Computational graph
    print("\n5. Computational Graph")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)
    a = x + 1
    b = a * 2
    c = b ** 2
    c.backward()

    print(f"x = {x.item()}")
    print(f"a = x + 1 = {a.item()}")
    print(f"b = a * 2 = {b.item()}")
    print(f"c = b^2 = {c.item()}")
    print(f"\ndc/dx = {x.grad.item()}")
    # Chain rule: dc/dx = dc/db * db/da * da/dx = 2b * 2 * 1 = 4b = 4*6 = 24

    # 6. Detaching from graph
    print("\n6. Detaching from Graph")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    y_detached = y.detach()  # Remove from computational graph

    print(f"y.requires_grad = {y.requires_grad}")
    print(f"y_detached.requires_grad = {y_detached.requires_grad}")

    # Can't compute gradients through detached tensor
    z = y_detached ** 2
    print(f"z.requires_grad = {z.requires_grad}")

    # 7. Context managers
    print("\n7. Context Managers")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)

    # torch.no_grad() - disable gradient tracking
    with torch.no_grad():
        y = x ** 2
        print(f"Inside no_grad: y.requires_grad = {y.requires_grad}")

    # torch.set_grad_enabled() - conditionally enable/disable
    is_train = False
    with torch.set_grad_enabled(is_train):
        y = x ** 2
        print(f"With set_grad_enabled({is_train}): y.requires_grad = {y.requires_grad}")

    # 8. Gradient checkpointing concept
    print("\n8. Higher-order Gradients")
    print("-" * 60)

    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 3

    # First derivative
    grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"dy/dx = 3x^2 = {grad_y}")

    # Second derivative
    grad2_y = torch.autograd.grad(grad_y, x)[0]
    print(f"d²y/dx² = 6x = {grad2_y}")

    # 9. Common pitfalls
    print("\n9. Common Pitfalls")
    print("-" * 60)

    print("Pitfall 1: Forgetting to zero gradients")
    x = torch.tensor([1.0], requires_grad=True)
    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"  Iteration {i+1}: x.grad = {x.grad.item()} (accumulating!)")
    print("  Solution: Call x.grad.zero_() or optimizer.zero_grad()")

    print("\nPitfall 2: Calling backward on non-scalar")
    try:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x ** 2
        y.backward()  # This will fail!
    except RuntimeError as e:
        print(f"  Error: {e}")
        print("  Solution: Provide gradient argument or sum to scalar")

    print("\nPitfall 3: In-place operations breaking gradients")
    try:
        x = torch.tensor([1.0], requires_grad=True)
        y = x ** 2
        x += 1  # In-place operation on leaf variable
        y.backward()
    except RuntimeError as e:
        print(f"  Error: {type(e).__name__}")
        print("  Solution: Avoid in-place ops on tensors with requires_grad=True")

    # 10. Practical example: computing Jacobian
    print("\n10. Practical Example: Jacobian Matrix")
    print("-" * 60)

    def f(x):
        """Function R^2 -> R^2: f(x) = [x1^2 + x2, x1 * x2]"""
        return torch.stack([x[0]**2 + x[1], x[0] * x[1]])

    x = torch.tensor([2.0, 3.0], requires_grad=True)

    # Compute Jacobian matrix
    jacobian = []
    for i in range(2):
        grad_output = torch.zeros(2)
        grad_output[i] = 1.0

        x.grad = None  # Clear previous gradients
        y = f(x)
        y.backward(gradient=grad_output, retain_graph=True)
        jacobian.append(x.grad.clone())

    jacobian = torch.stack(jacobian)
    print(f"x = {x.detach()}")
    print(f"Jacobian:\n{jacobian}")
    # For x=[2,3]: Jacobian = [[2*x1, 1], [x2, x1]] = [[4, 1], [3, 2]]

    print("\n" + "=" * 60)
    print("Lab 1.3 Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Set requires_grad=True to track operations")
    print("2. Call .backward() to compute gradients")
    print("3. Access gradients with .grad")
    print("4. Always zero gradients before new backward pass")
    print("5. Use torch.no_grad() for inference")


if __name__ == "__main__":
    main()
