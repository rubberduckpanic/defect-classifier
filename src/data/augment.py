"""Shared augmentation transform factory."""

from src.training.dataset import get_eval_transforms, get_train_transforms as build_train_transforms


def get_train_transforms(config: dict):
    """Return the configured training augmentation pipeline."""
    return build_train_transforms(config)


def get_val_transforms(config: dict):
    """Return the configured validation/test transform pipeline."""
    return get_eval_transforms(config)


if __name__ == "__main__":
    print("Train transforms:", get_train_transforms())
    print("Val transforms:", get_val_transforms())
