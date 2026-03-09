"""EfficientNet-B0 classifier used by the inference service."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class SkinLesionClassifier(nn.Module):
    """EfficientNet-B0 with a custom MLP head for 7-way classification."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        feature_dim = 1280

        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
