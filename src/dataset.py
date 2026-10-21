"""
DriveGuard AI — PyTorch Dataset & DataLoader Module
Provides custom PyTorch Dataset implementation with robust image loading
and multi-threaded DataLoader creation.
"""

import os
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import get_transforms


class DrowsinessDataset(Dataset):
    """
    Custom PyTorch Dataset for DriveGuard Driver Drowsiness Detection.
    """
    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        raw_path = record["filepath"]
        label = record["label"]
        rel_name = os.path.basename(raw_path)
        cls_folder = record["class_name"].replace(" ", "_")
        cached_path = os.path.join("data", "cache_128x128", cls_folder, rel_name)
        img_path = cached_path if os.path.exists(cached_path) else raw_path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to black image in case of I/O error
            image = Image.new("RGB", (128, 128), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, record["filename"]


def create_dataloaders(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    image_size=(128, 128),
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Instantiates train, validation, and test PyTorch DataLoaders.
    """
    train_transform = get_transforms(image_size=image_size, augment=True)
    eval_transform = get_transforms(image_size=image_size, augment=False)

    train_dataset = DrowsinessDataset(train_records, transform=train_transform)
    val_dataset = DrowsinessDataset(val_records, transform=eval_transform)
    test_dataset = DrowsinessDataset(test_records, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
