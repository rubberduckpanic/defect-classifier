"""Reusable model evaluation and MLflow experiment comparison."""

from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import yaml

from src.training.dataset import CastingDefectDataset, get_eval_transforms
from src.training.models import get_model


def evaluate_model(model_path: str, test_dir: str) -> dict:
    """
    Evaluate a saved model on test data.

    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_config = config.get("model", {"architecture": "resnet18", "num_classes": 2})
    model = get_model(
        architecture=model_config.get("architecture", "resnet18"),
        num_classes=model_config.get("num_classes", 2),
        pretrained=False,
        dropout=model_config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if not config:
        config = yaml.safe_load(Path("configs/train_config.yaml").read_text(encoding="utf-8"))
    dataset = CastingDefectDataset(test_dir, transform=get_eval_transforms(config))
    if not dataset:
        raise ValueError(f"No images found in {test_dir}")
    loader = DataLoader(dataset, batch_size=config["data"].get("batch_size", 32), shuffle=False)
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            predictions.extend(model(images).argmax(dim=1).numpy())
            labels.extend(batch_labels.numpy())

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "samples": len(labels),
    }


def compare_experiments(experiment_name: str) -> None:
    """Compare all runs in an MLflow experiment and report best model."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty or "metrics.test_accuracy" not in runs:
        return {"experiment": experiment_name, "runs": [], "best_run_id": None}
    runs = runs.dropna(subset=["metrics.test_accuracy"]).sort_values(
        "metrics.test_accuracy", ascending=False
    )
    results = []
    for _, run in runs.iterrows():
        results.append({
            "run_id": run.run_id,
            "test_accuracy": float(run["metrics.test_accuracy"]),
            "test_f1": float(run.get("metrics.test_f1", np.nan)),
            "architecture": run.get("params.architecture", "unknown"),
        })
    return {"experiment": experiment_name, "runs": results, "best_run_id": results[0]["run_id"]}


if __name__ == "__main__":
    print(evaluate_model("models/best_resnet18.pt", "data/splits/test"))
