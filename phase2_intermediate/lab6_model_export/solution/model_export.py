"""
Phase 2 Lab 6: Model Export and Optimization
Learn to export PyTorch models to TorchScript, ONNX, and optimize for inference
"""

import torch
import torch.nn as nn
import torch.onnx
import time
from pathlib import Path


class SimpleClassifier(nn.Module):
    """Simple classifier for export demonstration."""
    def __init__(self, input_dim: int = 100, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ConditionalModel(nn.Module):
    """Model with conditional logic for TorchScript."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, use_relu: bool = True):
        out = self.fc(x)
        if use_relu:
            out = torch.relu(out)
        return out


def demo_torchscript_tracing():
    """Demonstrate TorchScript via tracing."""
    print("=" * 70)
    print("DEMO: TorchScript - Tracing")
    print("=" * 70)

    # Create model
    model = SimpleClassifier()
    model.eval()

    # Create example input
    example_input = torch.randn(1, 100)

    print("Original model forward pass:")
    with torch.no_grad():
        output = model(example_input)
        print(f"  Output shape: {output.shape}")

    # Trace the model
    print("\nTracing model...")
    traced_model = torch.jit.trace(model, example_input)

    # Test traced model
    print("Traced model forward pass:")
    with torch.no_grad():
        traced_output = traced_model(example_input)
        print(f"  Output shape: {traced_output.shape}")

    # Verify outputs match
    print(f"  Outputs match: {torch.allclose(output, traced_output)}")

    # Save traced model
    save_path = Path("traced_model.pt")
    traced_model.save(str(save_path))
    print(f"\nTraced model saved to {save_path}")

    # Load and test
    loaded_model = torch.jit.load(str(save_path))
    with torch.no_grad():
        loaded_output = loaded_model(example_input)
        print(f"Loaded model works: {torch.allclose(output, loaded_output)}")

    # Cleanup
    save_path.unlink()

    print("=" * 70)


def demo_torchscript_scripting():
    """Demonstrate TorchScript via scripting."""
    print("\n" + "=" * 70)
    print("DEMO: TorchScript - Scripting")
    print("=" * 70)

    # Model with control flow
    model = ConditionalModel()
    model.eval()

    example_input = torch.randn(1, 10)

    print("Original model with ReLU:")
    with torch.no_grad():
        output_relu = model(example_input, use_relu=True)
        print(f"  Output: {output_relu[0, :5].tolist()[:5]}")

    print("\nOriginal model without ReLU:")
    with torch.no_grad():
        output_no_relu = model(example_input, use_relu=False)
        print(f"  Output: {output_no_relu[0, :5].tolist()[:5]}")

    # Script the model (preserves control flow)
    print("\nScripting model...")
    scripted_model = torch.jit.script(model)

    print("Scripted model with ReLU:")
    with torch.no_grad():
        scripted_relu = scripted_model(example_input, use_relu=True)
        print(f"  Matches original: {torch.allclose(output_relu, scripted_relu)}")

    print("Scripted model without ReLU:")
    with torch.no_grad():
        scripted_no_relu = scripted_model(example_input, use_relu=False)
        print(f"  Matches original: {torch.allclose(output_no_relu, scripted_no_relu)}")

    # View script code
    print("\nScripted model code:")
    print(scripted_model.code)

    print("=" * 70)


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "=" * 70)
    print("DEMO: ONNX Export")
    print("=" * 70)

    # Create model
    model = SimpleClassifier()
    model.eval()

    # Example input
    example_input = torch.randn(1, 100)
    example_output = model(example_input)

    # Export to ONNX
    onnx_path = Path("model.onnx")
    print(f"Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        example_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {onnx_path}")
    print(f"  File size: {onnx_path.stat().st_size / 1024:.2f} KB")

    # Try loading with ONNX (if available)
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        # Try ONNX Runtime (if available)
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            ort_inputs = {session.get_inputs()[0].name: example_input.numpy()}
            ort_output = session.run(None, ort_inputs)[0]

            print(f"✓ ONNX Runtime inference successful")
            print(f"  Outputs match PyTorch: {torch.allclose(example_output, torch.from_numpy(ort_output), atol=1e-5)}")
        except ImportError:
            print("  onnxruntime not installed (optional)")
    except ImportError:
        print("  onnx package not installed (optional)")

    # Cleanup
    onnx_path.unlink()

    print("=" * 70)


def demo_quantization():
    """Demonstrate dynamic quantization for inference speedup."""
    print("\n" + "=" * 70)
    print("DEMO: Dynamic Quantization")
    print("=" * 70)

    # Create larger model for visible speedup
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    model.eval()

    # Example input
    example_input = torch.randn(100, 512)

    # Benchmark FP32 model
    print("Benchmarking FP32 model...")
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(example_input)

        # Time
        start = time.time()
        for _ in range(100):
            _ = model(example_input)
        fp32_time = time.time() - start

    # Get model size
    fp32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024  # KB

    print(f"  Time: {fp32_time:.4f}s")
    print(f"  Size: {fp32_size:.2f} KB")

    # Dynamic quantization
    print("\nApplying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )

    # Benchmark quantized model
    print("Benchmarking quantized model...")
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = quantized_model(example_input)

        # Time
        start = time.time()
        for _ in range(100):
            _ = quantized_model(example_input)
        quant_time = time.time() - start

    # Get quantized model size
    quant_size = sum(
        p.numel() * p.element_size() if not hasattr(p, '_packed_params')
        else sum(x.numel() * x.element_size() for x in p._weight_bias())
        for p in quantized_model.parameters()
    ) / 1024

    print(f"  Time: {quant_time:.4f}s")
    print(f"  Size: {quant_size:.2f} KB")

    print(f"\nSpeedup: {fp32_time / quant_time:.2f}x")
    print(f"Size reduction: {fp32_size / quant_size:.2f}x smaller")

    # Verify accuracy
    with torch.no_grad():
        fp32_output = model(example_input)
        quant_output = quantized_model(example_input)
        max_diff = (fp32_output - quant_output).abs().max()
        print(f"Max difference: {max_diff:.6f}")

    print("=" * 70)


def demo_model_optimization_tips():
    """Print model optimization best practices."""
    print("\n" + "=" * 70)
    print("MODEL OPTIMIZATION BEST PRACTICES")
    print("=" * 70)

    practices = [
        ("1. Use TorchScript for deployment", "Removes Python overhead, enables C++ inference"),
        ("2. Export to ONNX for cross-platform", "Run on different frameworks/hardware"),
        ("3. Apply quantization", "8-bit precision: 4x smaller, 2-4x faster"),
        ("4. Use dynamic_axes in ONNX", "Support variable batch sizes"),
        ("5. Fuse operations", "BatchNorm + Conv fusion for speedup"),
        ("6. Prune unnecessary layers", "Remove weights close to zero"),
        ("7. Distill to smaller model", "Train small model to mimic large one"),
        ("8. Benchmark on target hardware", "Different optimizations for CPU/GPU/mobile"),
    ]

    for title, description in practices:
        print(f"\n{title}")
        print(f"  → {description}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)

    demo_torchscript_tracing()
    demo_torchscript_scripting()
    demo_onnx_export()
    demo_quantization()
    demo_model_optimization_tips()

    print("\n✓ Model export demonstrations complete!")
