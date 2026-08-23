"""
Module M3: Model Training with MLflow Tracking
Trains a single model configuration with full experiment logging.

All hyperparameters, metrics, and artifacts are tracked in MLflow for
reproducibility. Any experiment can be re-run from its logged configuration.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from src.training.dataset import get_dataloaders
from src.training.models import count_parameters, get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from config."""
    train_config = config["training"]
    
    if train_config["optimizer"] == "adam":
        return optim.Adam(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    elif train_config["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=train_config["learning_rate"],
            momentum=0.9,
            weight_decay=train_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")


def get_scheduler(optimizer: optim.Optimizer, config: dict, num_epochs: int):
    """Create learning rate scheduler from config."""
    scheduler_type = config["training"].get("scheduler", "cosine")
    
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(config: dict) -> Dict:
    """
    Full training loop with MLflow tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with training results and metrics
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    seed = config["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    model_config = config["model"]
    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False)
    )
    model = model.to(device)
    param_counts = count_parameters(model)
    logger.info(f"Model: {model_config['architecture']}, Parameters: {param_counts}")

    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    logger.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, config["training"]["epochs"])

    # MLflow tracking
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_config.get("experiment_name", "defect_classification"))

    best_val_acc = 0.0
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 5)
    results = {}

    with mlflow.start_run(run_name=f"{model_config['architecture']}_{int(time.time())}"):
        # Log parameters
        mlflow.log_params({
            "architecture": model_config["architecture"],
            "pretrained": model_config.get("pretrained", True),
            "num_classes": model_config["num_classes"],
            "dropout": model_config.get("dropout", 0.3),
            "freeze_backbone": model_config.get("freeze_backbone", False),
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "scheduler": config["training"].get("scheduler", "cosine"),
            "batch_size": config["data"]["batch_size"],
            "image_size": config["data"]["image_size"],
            "weight_decay": config["training"]["weight_decay"],
            "total_params": param_counts["total"],
            "trainable_params": param_counts["trainable"],
            "random_seed": seed,
        })

        # Training loop
        for epoch in range(config["training"]["epochs"]):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                best_model_path = model_dir / f"best_{model_config['architecture']}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "config": config,
                }, str(best_model_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation on test set
        checkpoint = torch.load(str(best_model_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        # Compute detailed metrics
        test_precision = precision_score(test_labels, test_preds, average="binary")
        test_recall = recall_score(test_labels, test_preds, average="binary")
        test_f1 = f1_score(test_labels, test_preds, average="binary")

        # Log final metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "best_val_accuracy": best_val_acc,
        })

        # Log classification report
        report = classification_report(
            test_labels, test_preds,
            target_names=["non_defective", "defective"]
        )
        logger.info(f"\nClassification Report:\n{report}")

        # Log model artifact
        if mlflow_config.get("log_models", True):
            try:
                mlflow.pytorch.log_model(
                    model, "model",
                    serialization_format="cloudpickle"
                )
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")
                # Model is still saved as checkpoint file

        # Log confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(report, "classification_report.txt")

        results = {
            "architecture": model_config["architecture"],
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "model_path": str(best_model_path),
            "run_id": mlflow.active_run().info.run_id,
        }

        logger.info(f"\nFinal Results: {results}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train defect classifier")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results = train_model(config)
    print(f"\nTraining complete. Results: {results}")
