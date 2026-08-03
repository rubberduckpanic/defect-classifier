"""PyTorch Dataset class for casting defect images."""

from torch.utils.data import Dataset


class DefectDataset(Dataset):
    """Custom Dataset for loading defect/ok images with transforms."""

    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Path to the processed image directory (with class subfolders).
            transform: Optional Albumentations or torchvision transform.
        """
        # TODO: Load file paths and labels from directory structure
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # List of (path, label) tuples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: Load image, apply transforms, return (image_tensor, label)
        raise NotImplementedError
