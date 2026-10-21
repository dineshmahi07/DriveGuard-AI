"""
DriveGuard AI — Training Engine
Features automatic device configuration, multi-threaded CPU/GPU optimization,
AdamW optimizer with Cosine Annealing, early stopping, best checkpointing,
and structured JSON metric logging.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import create_dataloaders
from src.models.custom_cnn import DriveGuardCNN
from src.models.mobilenet_model import DriveGuardMobileNetV2


def get_device() -> torch.device:
    """Detects available computing hardware and optimizes thread count"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        threads = min(16, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"[Device] Using CPU with {threads} PyTorch threads")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch and returns mean loss and accuracy"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates validation performance (loss, accuracy, macro-F1)"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    return val_loss, val_acc, val_f1


def train_model(
    model_name: str = "cnn",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 3,
    smoke_test: bool = False,
    manifest_path: str = "outputs/metrics/dataset_splits.json"
) -> Dict[str, Any]:
    """
    Main model training workflow:
    Loads dataset, initializes architecture, trains with early stopping,
    checkpoints the best weights, and writes history.
    """
    device = get_device()
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_records = manifest["train_records"]
    val_records = manifest["val_records"]
    test_records = manifest["test_records"]

    if smoke_test:
        print("[Smoke Test] Subsetting to 64 train and 32 val samples for verification...")
        train_records = train_records[:64]
        val_records = val_records[:32]
        epochs = 1

    loaders = create_dataloaders(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        batch_size=batch_size
    )

    # Architecture instantiation
    if model_name.lower() in ["cnn", "custom_cnn", "driveguard_cnn"]:
        model = DriveGuardCNN(num_classes=2).to(device)
        model_tag = "custom_cnn"
        ckpt_filename = "best_driveguard_cnn.pt"
    elif model_name.lower() in ["mobilenet", "mobilenet_v2", "driveguard_mobilenet"]:
        # Pretrained MobileNetV2 fine-tuning
        model = DriveGuardMobileNetV2(num_classes=2, pretrained=True).to(device)
        model_tag = "mobilenet_v2"
        ckpt_filename = "best_mobilenet_v2.pt"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"\n==================================================")
    print(f"Starting Training: {model_tag.upper()} ({model.count_parameters():,} parameters)")
    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}, Epochs: {epochs}")
    print(f"==================================================")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / ckpt_filename

    history = {
        "model_name": model_tag,
        "parameters": model.count_parameters(),
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epoch_times": [],
        "best_epoch": 0,
        "best_val_f1": 0.0,
        "total_training_time_sec": 0.0
    }

    best_val_f1 = -1.0
    epochs_no_improve = 0
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["epoch_times"].append(round(epoch_duration, 2))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1*100:.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            history["best_epoch"] = epoch
            history["best_val_f1"] = round(best_val_f1, 4)
            epochs_no_improve = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "model_name": model_tag,
                "num_classes": 2,
                "image_size": [128, 128]
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val F1: {best_val_f1*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not smoke_test:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    history["total_training_time_sec"] = round(time.time() - start_total_time, 2)
    
    # Save training history
    history_path = Path("outputs/metrics") / f"history_{model_tag}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining for {model_tag} completed in {history['total_training_time_sec']:.1f}s. History saved to {history_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriveGuard AI Training Script")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenet"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        smoke_test=args.smoke_test
    )
