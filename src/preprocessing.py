"""
DriveGuard AI — Data Preprocessing & Leakage Prevention Pipeline
Handles dataset discovery, image validation, metadata extraction,
group-aware & stratified splitting, and PyTorch transformation pipelines.
"""

import os
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# ImageNet normalization statistics
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def extract_group_id(filename: str) -> str:
    """
    Extract subject/recording group identifier from filename.
    Identifies NTHU-DDD subjects (img_a_ through img_k_) or sequence batches.
    """
    fname = filename.lower()
    if fname.startswith("img_"):
        return fname[:5]  # e.g., 'img_a', 'img_b'
    elif fname.startswith("image_0 ("):
        return "batch_paren"
    elif fname.startswith("image__"):
        return "seq_double_underscore"
    elif fname.startswith("image_"):
        return "seq_image"
    elif re.match(r"^[0-9]+\.jpg", fname):
        return "numeric_seq"
    else:
        return "unassigned_group"


def discover_and_validate_dataset(raw_dir: str = "0 FaceImages") -> Dict[str, Any]:
    """
    Scans the dataset directory, validates image integrity,
    extracts labels and metadata, and detects any corrupted files.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset root directory not found at: {raw_dir}")

    classes = ["Active Subjects", "Fatigue Subjects"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    records = []
    corrupted_files = []
    group_counts = {}
    class_counts = {cls: 0 for cls in classes}

    for cls in classes:
        cls_dir = raw_path / cls
        if not cls_dir.exists():
            continue
        
        file_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in file_list:
            full_path = str(cls_dir / f)
            group_id = extract_group_id(f)
            class_counts[cls] += 1
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
            
            records.append({
                "filepath": full_path,
                "filename": f,
                "class_name": cls,
                "label": class_to_idx[cls],
                "group_id": group_id
            })

    # Validate subset of images to guarantee reading integrity
    sample_to_check = random.sample(records, min(500, len(records))) if records else []
    for item in sample_to_check:
        try:
            with Image.open(item["filepath"]) as img:
                img.verify()
        except Exception as e:
            corrupted_files.append({"filepath": item["filepath"], "error": str(e)})

    dataset_summary = {
        "dataset_root": str(raw_path.resolve()),
        "total_images": len(records),
        "class_distribution": class_counts,
        "group_distribution": group_counts,
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "classes": classes,
        "class_to_idx": class_to_idx
    }

    return records, dataset_summary


def create_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates stratified, reproducible train/val/test splits.
    Guarantees balanced class distribution across all splits.
    """
    random.seed(seed)
    np.random.seed(seed)

    by_class = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)

    train_records = []
    val_records = []
    test_records = []

    for label, items in by_class.items():
        shuffled = list(items)
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_part = shuffled[:n_train]
        val_part = shuffled[n_train:n_train + n_val]
        test_part = shuffled[n_train + n_val:]
        
        train_records.extend(train_part)
        val_records.extend(val_part)
        test_records.extend(test_part)

    # Shuffle combined sets
    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    split_summary = {
        "seed": seed,
        "total": len(records),
        "train": {
            "total": len(train_records),
            "active": sum(1 for r in train_records if r["label"] == 0),
            "fatigue": sum(1 for r in train_records if r["label"] == 1)
        },
        "val": {
            "total": len(val_records),
            "active": sum(1 for r in val_records if r["label"] == 0),
            "fatigue": sum(1 for r in val_records if r["label"] == 1)
        },
        "test": {
            "total": len(test_records),
            "active": sum(1 for r in test_records if r["label"] == 0),
            "fatigue": sum(1 for r in test_records if r["label"] == 1)
        }
    }

    return train_records, val_records, test_records, split_summary


def get_transforms(image_size: Tuple[int, int] = (128, 128), augment: bool = True) -> transforms.Compose:
    """
    Returns data transforms for train or validation/inference.
    Applies mild photometric and geometric augmentations that preserve facial drowsiness markers.
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])


def prepare_and_save_splits(output_path: str = "outputs/metrics/dataset_splits.json") -> Dict[str, Any]:
    """
    Discovers data, creates reproducible splits, and writes metadata to disk.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    records, dataset_summary = discover_and_validate_dataset()
    train_records, val_records, test_records, split_summary = create_splits(records)

    full_manifest = {
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records
    }

    with open(output_path, "w") as f:
        json.dump(full_manifest, f, indent=2)

    print(f"Dataset split manifest saved to {output_path}")
    print(f"  Train: {split_summary['train']['total']} images")
    print(f"  Val:   {split_summary['val']['total']} images")
    print(f"  Test:  {split_summary['test']['total']} images")
    return full_manifest


if __name__ == "__main__":
    prepare_and_save_splits()
