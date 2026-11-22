"""Hierarchical model: a species gate plus one condition head per crop.

Why hierarchical (per the brief): a single flat classifier over all 22
conditions can confuse labels across crops - predicting "Grape Black Rot"
for a potato leaf. Gating on species first makes that structurally
impossible, because each condition head only knows its own crop's classes.

The backbone is shared and, by default, FROZEN. With no GPU available,
freezing lets features be computed once and cached, after which the linear
heads train in seconds instead of hours. Pass finetune=True to unfreeze for
a GPU run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

BACKBONES = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
    "efficientnet_b0":    (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "resnet50":           (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


def build_backbone(name: str, finetune: bool = False) -> tuple[nn.Module, int]:
    """Return (feature extractor producing a flat embedding, embedding dim)."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    ctor, weights, dim = BACKBONES[name]
    net = ctor(weights=weights)

    if name == "resnet50":
        net.fc = nn.Identity()
        feat = net
    else:
        # mobilenet/efficientnet: features -> pool -> flatten
        feat = nn.Sequential(
            net.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    if not finetune:
        for p in feat.parameters():
            p.requires_grad = False
        feat.eval()
    return feat, dim


class LinearHead(nn.Module):
    """Dropout + linear. Deliberately shallow: with a frozen backbone and a
    few hundred images in the smallest classes, a deeper head overfits."""

    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalClassifier(nn.Module):
    """Species gate + per-crop condition heads over a shared backbone."""

    def __init__(
        self,
        backbone: str,
        crops: list[str],
        conditions_per_crop: dict[str, int],
        finetune: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.crops = list(crops)
        self.finetune = finetune
        self.features, self.dim = build_backbone(backbone, finetune)
        self.species_head = LinearHead(self.dim, len(self.crops), dropout)
        self.condition_heads = nn.ModuleDict({
            crop: LinearHead(self.dim, n, dropout)
            for crop, n in conditions_per_crop.items()
        })

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.finetune:
            return self.features(x)
        with torch.no_grad():
            return self.features(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.embed(x)
        return self.heads_from_embedding(z)

    def heads_from_embedding(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"embedding": z, "species": self.species_head(z)}
        for crop, head in self.condition_heads.items():
            out[f"condition_{crop}"] = head(z)
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
