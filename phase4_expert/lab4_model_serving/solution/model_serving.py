"""
Production Model Serving with TorchServe
Deploy PyTorch models with low latency and high throughput
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List


# Example model for serving
class ProductionModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class OptimizedModel:
    """Model with various optimizations for serving"""

    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def quantize_dynamic(self):
        """Dynamic quantization (INT8) - 4x smaller, faster"""
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        print("✓ Applied dynamic quantization (INT8)")

    def to_torchscript(self):
        """Convert to TorchScript for faster inference"""
        example_input = torch.randn(1, 100).to(self.device)
        self.model = torch.jit.trace(self.model, example_input)
        print("✓ Converted to TorchScript")

    def to_onnx(self, path='model.onnx'):
        """Export to ONNX for cross-platform deployment"""
        example_input = torch.randn(1, 100).to(self.device)
        torch.onnx.export(
            self.model,
            example_input,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✓ Exported to ONNX: {path}")

    def benchmark(self, batch_sizes=[1, 8, 32, 128], num_iterations=100):
        """Benchmark inference latency"""
        print("\n" + "=" * 60)
        print("Latency Benchmark")
        print("=" * 60)

        results = []

        with torch.no_grad():
            for batch_size in batch_sizes:
                inputs = torch.randn(batch_size, 100).to(self.device)

                # Warmup
                for _ in range(10):
                    _ = self.model(inputs)

                # Benchmark
                start = time.time()
                for _ in range(num_iterations):
                    _ = self.model(inputs)
                elapsed = time.time() - start

                latency_ms = (elapsed / num_iterations) * 1000
                throughput = (batch_size * num_iterations) / elapsed

                print(f"Batch {batch_size:3d}: {latency_ms:6.2f}ms/batch, {throughput:8.0f} samples/sec")

                results.append({
                    'batch_size': batch_size,
                    'latency_ms': latency_ms,
                    'throughput': throughput
                })

        return results


# TorchServe handler example
TORCHSERVE_HANDLER = '''
"""
Custom TorchServe handler for production serving
Save as: model_handler.py
"""

import torch
import logging

logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Custom handler for TorchServe

    Methods:
    - initialize: Load model
    - preprocess: Prepare input
    - inference: Run model
    - postprocess: Format output
    """

    def __init__(self):
        self.model = None
        self.initialized = False

    def initialize(self, context):
        """Load model and initialize"""
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        model_path = f"{model_dir}/model.pt"
        self.model = torch.jit.load(model_path)
        self.model.eval()

        self.initialized = True
        logger.info("Model initialized successfully")

    def preprocess(self, requests):
        """Preprocess input data"""
        inputs = []

        for request in requests:
            data = request.get("data") or request.get("body")

            # Parse input (example: JSON with 'features' field)
            if isinstance(data, dict):
                features = torch.tensor(data['features'], dtype=torch.float32)
            else:
                features = torch.tensor(data, dtype=torch.float32)

            inputs.append(features)

        # Batch inputs
        inputs = torch.stack(inputs)
        return inputs

    def inference(self, inputs):
        """Run inference"""
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs):
        """Format outputs"""
        predictions = outputs.tolist()
        return predictions
'''


if __name__ == "__main__":
    print("=" * 60)
    print("Production Model Serving")
    print("=" * 60)

    # Create model
    model = ProductionModel(input_dim=100, hidden_dim=256, output_dim=10)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Baseline benchmark
    print("\n1. Baseline Model (FP32, Eager Mode)")
    print("-" * 60)

    baseline = OptimizedModel(model, device='cpu')
    baseline_results = baseline.benchmark()

    # Optimization 1: TorchScript
    print("\n2. TorchScript Optimization")
    print("-" * 60)

    ts_model = ProductionModel(input_dim=100, hidden_dim=256, output_dim=10)
    ts_opt = OptimizedModel(ts_model, device='cpu')
    ts_opt.to_torchscript()
    ts_results = ts_opt.benchmark()

    # Optimization 2: Quantization
    print("\n3. Quantization (INT8)")
    print("-" * 60)

    quant_model = ProductionModel(input_dim=100, hidden_dim=256, output_dim=10)
    quant_opt = OptimizedModel(quant_model, device='cpu')
    quant_opt.quantize_dynamic()
    quant_results = quant_opt.benchmark()

    # Summary
    print("\n" + "=" * 60)
    print("Optimization Summary (Batch Size 32)")
    print("=" * 60)

    baseline_latency = [r for r in baseline_results if r['batch_size'] == 32][0]['latency_ms']
    ts_latency = [r for r in ts_results if r['batch_size'] == 32][0]['latency_ms']
    quant_latency = [r for r in quant_results if r['batch_size'] == 32][0]['latency_ms']

    print(f"Baseline:     {baseline_latency:.2f}ms")
    print(f"TorchScript:  {ts_latency:.2f}ms ({baseline_latency/ts_latency:.2f}x faster)")
    print(f"Quantized:    {quant_latency:.2f}ms ({baseline_latency/quant_latency:.2f}x faster)")

    # Export to ONNX
    print("\n" + "=" * 60)
    print("ONNX Export")
    print("=" * 60)

    onnx_model = ProductionModel(input_dim=100, hidden_dim=256, output_dim=10)
    onnx_opt = OptimizedModel(onnx_model, device='cpu')
    onnx_opt.to_onnx('model.onnx')

    print("\n" + "=" * 60)
    print("TorchServe Deployment")
    print("=" * 60)

    print("\n1. Package model:")
    print("   torch-model-archiver --model-name my_model \\")
    print("     --version 1.0 \\")
    print("     --serialized-file model.pt \\")
    print("     --handler model_handler.py")

    print("\n2. Start TorchServe:")
    print("   torchserve --start --model-store model_store \\")
    print("     --models my_model=my_model.mar")

    print("\n3. Make prediction:")
    print("   curl -X POST http://localhost:8080/predictions/my_model \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"features\": [...]}'")

    print("\n" + "=" * 60)
    print("Production Best Practices:")
    print("=" * 60)
    print("✓ Use TorchScript or ONNX for faster inference")
    print("✓ Apply quantization (INT8) for 4x compression")
    print("✓ Enable dynamic batching for higher throughput")
    print("✓ Monitor latency (p50, p95, p99)")
    print("✓ Set up autoscaling based on QPS")
    print("✓ Use model versioning for safe rollouts")
    print("✓ Log predictions for debugging")
    print("✓ A/B test new models before full deployment")
