"""Training script with MLflow experiment tracking.

Trains models, logs metrics/params/artifacts to MLflow.
"""

import argparse


def train(config_path: str) -> None:
    """
    Main training loop.

    Steps:
        1. Load config from YAML
        2. Initialize dataset and dataloaders
        3. Initialize model, optimizer, scheduler
        4. Training loop with MLflow logging
        5. Save best model checkpoint
    """
    # TODO: Implement full training pipeline
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train defect classifier")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to training configuration YAML"
    )
    args = parser.parse_args()
    train(args.config)
