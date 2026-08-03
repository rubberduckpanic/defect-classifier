"""Model evaluation and comparison module.

Computes metrics, generates confusion matrices, and compares experiments.
"""


def evaluate_model(model_path: str, test_dir: str) -> dict:
    """
    Evaluate a saved model on test data.

    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix.
    """
    # TODO: Load model, run inference on test set, compute metrics
    raise NotImplementedError


def compare_experiments(experiment_name: str) -> None:
    """Compare all runs in an MLflow experiment and report best model."""
    # TODO: Query MLflow, rank by metric, output comparison table
    raise NotImplementedError


if __name__ == "__main__":
    evaluate_model("models/best_model.pt", "data/processed/test")
