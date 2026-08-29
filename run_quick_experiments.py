"""Quick experiment runner: trains EfficientNet-B0 for a 3rd tracked experiment."""
import yaml
from copy import deepcopy
from src.training.train import train_model

if __name__ == "__main__":
    with open("configs/train_config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # Experiment: EfficientNet-B0
    config = deepcopy(base_config)
    config["model"]["architecture"] = "efficientnet_b0"
    config["model"]["pretrained"] = True
    config["training"]["epochs"] = 3
    config["training"]["learning_rate"] = 0.0005
    config["data"]["num_workers"] = 0  # Windows safe

    print("=" * 50)
    print("EXPERIMENT: EfficientNet-B0 (3 epochs)")
    print("=" * 50)

    results = train_model(config)
    print(f"\nEfficientNet-B0 Results:")
    print(f"  Test Accuracy : {results['test_accuracy']:.4f}")
    print(f"  Test F1       : {results['test_f1']:.4f}")
    print(f"  MLflow Run ID : {results['run_id']}")
    print("\n3 experiments now tracked: ResNet18 + CNN Baseline + EfficientNet-B0")
