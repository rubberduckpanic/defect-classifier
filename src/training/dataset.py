"""
Module M3: PyTorch Dataset & DataLoaders
Custom dataset class with augmentation transforms for training.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CastingDefectDataset(Dataset):
    """
    PyTorch Dataset for Casting Defect images.
    
    Expects directory structure:
        root_dir/
            defective/
                img1.png, img2.png, ...
            non_defective/
                img1.png, img2.png, ...
    
    Labels: 0 = non_defective, 1 = defective
    """

    CLASS_NAMES = ["ok_front", "def_front"]
    CLASS_TO_IDX = {"ok_front": 0, "def_front": 1}

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

        for class_name in self.CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            label = self.CLASS_TO_IDX[class_name]
            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    self.samples.append(img_file)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(config: dict) -> transforms.Compose:
    """
    Training transforms with data augmentation.
    
    Augmentation strategy justification:
    - HorizontalFlip: Defects can appear on either side
    - Rotation: Production line orientation varies slightly
    - ColorJitter: Simulates lighting variation on factory floor
    - No vertical flip: Products maintain consistent vertical orientation
    """
    aug_config = config.get("augmentation", {})
    
    transform_list = [
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
    ]

    if aug_config.get("horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug_config.get("rotation_degrees", 0) > 0:
        transform_list.append(
            transforms.RandomRotation(aug_config["rotation_degrees"])
        )

    brightness = aug_config.get("brightness_jitter", 0.2)
    contrast = aug_config.get("contrast_jitter", 0.2)
    if brightness > 0 or contrast > 0:
        transform_list.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug_config.get("normalize_std", [0.229, 0.224, 0.225])
        ),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(config: dict) -> transforms.Compose:
    """Evaluation transforms (no augmentation, only resize + normalize)."""
    aug_config = config.get("augmentation", {})
    
    return transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=aug_config.get("normalize_std", [0.229, 0.224, 0.225])
        ),
    ])


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config["data"]
    splits_dir = data_config["splits_dir"]

    train_dataset = CastingDefectDataset(
        root_dir=f"{splits_dir}/train",
        transform=get_train_transforms(config)
    )
    val_dataset = CastingDefectDataset(
        root_dir=f"{splits_dir}/val",
        transform=get_eval_transforms(config)
    )
    test_dataset = CastingDefectDataset(
        root_dir=f"{splits_dir}/test",
        transform=get_eval_transforms(config)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
