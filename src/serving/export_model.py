"""Backward-compatible command wrapper for model export."""

from src.training.export_model import (
	benchmark_inference,
	export_to_onnx,
	export_to_torchscript,
)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Export model for production")
	parser.add_argument("--checkpoint", default="models/best_resnet18.pt")
	parser.add_argument("--format", choices=["torchscript", "onnx", "both"], default="torchscript")
	parser.add_argument("--benchmark", action="store_true")
	args = parser.parse_args()

	if args.format in ("torchscript", "both"):
		export_to_torchscript(args.checkpoint)
	if args.format in ("onnx", "both"):
		export_to_onnx(args.checkpoint)
	if args.benchmark:
		benchmark_inference(args.checkpoint)
