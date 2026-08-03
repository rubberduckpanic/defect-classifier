"""Data augmentation module.

Applies augmentation transforms to increase training data diversity.
"""


def get_train_transforms():
    """Return Albumentations compose pipeline for training augmentation."""
    # TODO: Define augmentation pipeline (flips, rotations, brightness, etc.)
    raise NotImplementedError


def get_val_transforms():
    """Return minimal transforms for validation/test (resize + normalize only)."""
    # TODO: Define validation transforms
    raise NotImplementedError


if __name__ == "__main__":
    print("Train transforms:", get_train_transforms())
    print("Val transforms:", get_val_transforms())
