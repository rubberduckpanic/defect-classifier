"""
Module M3: Model Architectures
Defines model architectures for defect classification.

Models:
1. CNN Baseline — Custom 4-layer CNN (no pretrained weights)
2. ResNet18 — Transfer learning with ImageNet weights
3. EfficientNet-B0 — Transfer learning with ImageNet weights

Design Justification:
- CNN Baseline serves as a lower bound to demonstrate transfer learning benefits
- ResNet18 balances accuracy and inference speed (suitable for production)
- EfficientNet-B0 offers best accuracy-per-FLOP (compound scaling)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBaseline(nn.Module):
    """
    Custom 4-layer CNN for binary classification.
    
    Architecture: Conv→BN→ReLU→Pool (x4) → FC layers
    Serves as baseline to justify transfer learning approach.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18Classifier(nn.Module):
    """
    ResNet18 with transfer learning for binary defect classification.
    
    Strategy: Replace final FC layer, optionally freeze backbone for fine-tuning.
    Justification: ResNet18 is lightweight (~11M params), well-suited for 
    binary classification with limited data. Skip connections help with 
    gradient flow during fine-tuning.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained ResNet18 when available; support offline training.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception as error:
            if not pretrained:
                raise
            print(f"Pretrained ResNet18 unavailable ({error}); using random initialization.")
            self.backbone = models.resnet18(weights=None)

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with transfer learning for defect classification.
    
    Justification: EfficientNet uses compound scaling (depth, width, resolution)
    for optimal accuracy/efficiency tradeoff. B0 variant is lightweight enough
    for production serving while offering superior feature extraction.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained EfficientNet-B0 when available; support offline training.
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception as error:
            if not pretrained:
                raise
            print(f"Pretrained EfficientNet-B0 unavailable ({error}); using random initialization.")
            self.backbone = models.efficientnet_b0(weights=None)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(architecture: str, **kwargs) -> nn.Module:
    """
    Factory function to create model by name.
    
    Args:
        architecture: One of "cnn_baseline", "resnet18", "efficientnet_b0"
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    models_map = {
        "cnn_baseline": CNNBaseline,
        "resnet18": ResNet18Classifier,
        "efficientnet_b0": EfficientNetB0Classifier,
    }

    if architecture not in models_map:
        raise ValueError(f"Unknown architecture: {architecture}. "
                         f"Choose from: {list(models_map.keys())}")

    # Filter kwargs based on what each model accepts
    model_class = models_map[architecture]
    if architecture == "cnn_baseline":
        # CNNBaseline only accepts num_classes and dropout
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("num_classes", "dropout")
        }
    else:
        filtered_kwargs = kwargs

    return model_class(**filtered_kwargs)


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
