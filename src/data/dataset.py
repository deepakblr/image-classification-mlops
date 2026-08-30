"""PyTorch datasets and transforms with augmentation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config


def train_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


class CatsDogsDataset(Dataset):
    def __init__(self, records: list[dict], transform=None, root: Path | None = None):
        self.records = records
        self.transform = transform
        self.root = root or config.PROJECT_ROOT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = self.root / rec["path"]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(rec["label"])
        return image, label


def build_dataloaders(
    splits: dict[str, list[dict]],
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(
            CatsDogsDataset(splits["train"], transform=train_transforms()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            CatsDogsDataset(splits["val"], transform=eval_transforms()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            CatsDogsDataset(splits["test"], transform=eval_transforms()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders
