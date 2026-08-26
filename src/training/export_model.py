"""
Module M4: Model Export & Serialization
Exports trained model to TorchScript and ONNX for production deployment.

Design Decision:
- TorchScript: Primary format for PyTorch-native serving (no Python dependency at runtime)
- ONNX: Secondary format for cross-framework compatibility and hardware acceleration
"""

import logging
from pathlib import Path

import torch
import yaml

from src.training.models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_to_torchscript(
    checkpoint_path: str,
    output_path: str = "models/model_scripted.pt",
    input_size: int = 224
) -> str:
    """
    Export model to TorchScript format using tracing.
    
    TorchScript enables:
    - Deployment without Python runtime
    - JIT optimizations
    - Consistent inference behavior
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Output path for scripted model
        input_size: Expected input dimension
        
    Returns:
        Path to exported model
    """
    device = torch.device("cpu")  # Export on CPU for portability

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Recreate model
    model = get_model(
        architecture=model_config.get("architecture", "resnet18"),
        num_classes=model_config.get("num_classes", 2),
        pretrained=False,
        dropout=model_config.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Trace the model with example input
    example_input = torch.randn(1, 3, input_size, input_size)
    
    try:
        traced_model = torch.jit.trace(model, example_input)
        
        # Verify traced model produces same output
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            
        diff = (original_output - traced_output).abs().max().item()
        assert diff < 1e-5, f"Output mismatch: max diff = {diff}"
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        traced_model.save(str(output_file))
        
        logger.info(f"TorchScript model saved to {output_path} (diff={diff:.2e})")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        raise


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "models/model.onnx",
    input_size: int = 224
) -> str:
    """
    Export model to ONNX format.
    
    ONNX enables:
    - Cross-framework deployment (TensorRT, OpenVINO, etc.)
    - Hardware-specific optimization
    - Broader runtime support
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Output path for ONNX model
        input_size: Expected input dimension
        
    Returns:
        Path to exported model
    """
    device = torch.device("cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    # Recreate model
    model = get_model(
        architecture=model_config.get("architecture", "resnet18"),
        num_classes=model_config.get("num_classes", 2),
        pretrained=False,
        dropout=model_config.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export
    example_input = torch.randn(1, 3, input_size, input_size)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example_input,
        str(output_file),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    logger.info(f"ONNX model saved to {output_path}")
    return str(output_file)


def benchmark_inference(
    checkpoint_path: str,
    num_iterations: int = 100,
    input_size: int = 224
) -> dict:
    """
    Benchmark model inference latency.
    
    Reports:
    - Mean latency
    - P50, P95, P99 latency
    - Throughput (images/second)
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model = get_model(
        architecture=model_config.get("architecture", "resnet18"),
        num_classes=model_config.get("num_classes", 2),
        pretrained=False,
        dropout=model_config.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Warmup
    example_input = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(example_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(example_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    import numpy as np
    latencies = np.array(latencies)

    results = {
        "device": str(device),
        "num_iterations": num_iterations,
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_imgs_per_sec": float(1000.0 / latencies.mean()),
    }

    logger.info(f"Benchmark results: {results}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model for production")
    parser.add_argument("--checkpoint", type=str, default="models/best_resnet18.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--format", type=str, choices=["torchscript", "onnx", "both"],
                        default="torchscript", help="Export format")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    args = parser.parse_args()

    if args.format in ("torchscript", "both"):
        export_to_torchscript(args.checkpoint)

    if args.format in ("onnx", "both"):
        export_to_onnx(args.checkpoint)

    if args.benchmark:
        benchmark_inference(args.checkpoint)
