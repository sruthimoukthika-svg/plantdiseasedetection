"""Manifest-driven dataset and the canonical preprocessing transforms.

The transforms here are the single source of truth. The FastAPI backend
imports the same `eval_transform` so inference preprocessing cannot drift
from training preprocessing - a classic source of silent accuracy loss.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
# ImageNet statistics; the backbones are ImageNet-pretrained.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def eval_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic path used for val/test AND for production inference."""
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmented path for training images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@dataclass
class Row:
    path: str
    split: str
    crop: str
    condition: str
    kind: str
    source_dataset: str
    raw_label: str
    dhash: str
    dup_group: str


def load_manifest(manifest: Path) -> list[Row]:
    with Path(manifest).open() as fh:
        return [Row(**r) for r in csv.DictReader(fh)]


class LeafDataset(Dataset):
    """Serves (image_tensor, crop_idx, condition_idx) from the manifest.

    `condition_idx` is indexed WITHIN the row's crop, matching the
    hierarchical design: a species gate plus one condition head per crop.
    """

    def __init__(
        self,
        rows: list[Row],
        image_root: Path,
        crop_index: dict[str, int],
        condition_index: dict[str, dict[str, int]],
        transform=None,
    ) -> None:
        self.rows = rows
        self.image_root = Path(image_root)
        self.crop_index = crop_index
        self.condition_index = condition_index
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        im = Image.open(self.image_root / r.path).convert("RGB")
        x = self.transform(im)
        return (
            x,
            torch.tensor(self.crop_index[r.crop], dtype=torch.long),
            torch.tensor(self.condition_index[r.crop][r.condition], dtype=torch.long),
        )
