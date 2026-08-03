"""Image preprocessing module.

Handles resizing, normalization, and format standardization.
"""


def preprocess_images(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    target_size: tuple = (224, 224),
) -> None:
    """Resize, normalize, and save preprocessed images."""
    # TODO: Implement preprocessing pipeline
    raise NotImplementedError


if __name__ == "__main__":
    preprocess_images()
