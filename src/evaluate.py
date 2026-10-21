"""
DriveGuard AI — Model Evaluation & Comparison Pipeline
Evaluates DriveGuardCNN and DriveGuardMobileNetV2 on the held-out test set (1,368 images).
Computes Accuracy, Balanced Accuracy, Precision, Recall, Macro F1, Per-Class Metrics,
Confusion Matrices, ROC-AUC, FP/FN counts, inference latencies, parameter counts, and model sizes.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from src.dataset import create_dataloaders
from src.models.custom_cnn import DriveGuardCNN
from src.models.mobilenet_model import DriveGuardMobileNetV2


def load_model_from_checkpoint(model_type: str, ckpt_path: str, device: torch.device):
    """Loads model architecture and weights from checkpoint file"""
    if model_type == "custom_cnn":
        model = DriveGuardCNN(num_classes=2)
    elif model_type == "mobilenet_v2":
        model = DriveGuardMobileNetV2(num_classes=2, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate_single_model(
    model_type: str,
    ckpt_path: str,
    test_loader,
    device: torch.device
) -> Dict[str, Any]:
    """Runs complete evaluation protocol on the held-out test set"""
    model, ckpt = load_model_from_checkpoint(model_type, ckpt_path, device)
    
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    num_params = model.count_parameters()

    all_targets = []
    all_preds = []
    all_probs = []
    all_filenames = []

    # Measure inference latency over test set
    latencies = []
    
    with torch.no_grad():
        for images, targets, filenames in test_loader:
            images = images.to(device)
            
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / images.size(0))

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_filenames.extend(filenames)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_latency_ms = float(np.mean(latencies) * 1000.0)

    acc = float(accuracy_score(all_targets, all_preds))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="macro"))
    rec = float(recall_score(all_targets, all_preds, average="macro"))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    roc_auc = float(roc_auc_score(all_targets, all_probs))

    # Per-class metrics (0: Active, 1: Fatigue)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    active_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    active_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    active_f1 = float(2 * active_prec * active_rec / (active_prec + active_rec)) if (active_prec + active_rec) > 0 else 0.0

    fatigue_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fatigue_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fatigue_f1 = float(2 * fatigue_prec * fatigue_rec / (fatigue_prec + fatigue_rec)) if (fatigue_prec + fatigue_rec) > 0 else 0.0

    cls_report = classification_report(
        all_targets, all_preds,
        target_names=["Active Subjects", "Fatigue Subjects"],
        output_dict=True
    )

    # Identify misclassified examples
    misclassified = []
    for idx in range(len(all_targets)):
        if all_targets[idx] != all_preds[idx]:
            misclassified.append({
                "filename": all_filenames[idx],
                "true_label": int(all_targets[idx]),
                "predicted_label": int(all_preds[idx]),
                "predicted_prob_fatigue": float(all_probs[idx])
            })

    results = {
        "model_type": model_type,
        "checkpoint_path": ckpt_path,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "trainable_parameters": num_params,
        "best_epoch_in_training": ckpt.get("epoch", 0),
        "test_samples_total": len(all_targets),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "true_negatives_active": tn,
        "false_positives_fatigue_alarm": fp,
        "false_negatives_missed_drowsiness": fn,
        "true_positives_detected_drowsiness": tp,
        "per_class": {
            "Active Subjects": {
                "precision": round(active_prec, 4),
                "recall": round(active_rec, 4),
                "f1_score": round(active_f1, 4),
                "support": tn + fp
            },
            "Fatigue Subjects": {
                "precision": round(fatigue_prec, 4),
                "recall": round(fatigue_rec, 4),
                "f1_score": round(fatigue_f1, 4),
                "support": tp + fn
            }
        },
        "classification_report": cls_report,
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
        "misclassified_count": len(misclassified),
        "sample_misclassified": misclassified[:15],
        "test_predictions": {
            "targets": all_targets.tolist(),
            "preds": all_preds.tolist(),
            "probs": all_probs.tolist(),
            "filenames": all_filenames
        }
    }

    return results


def run_full_evaluation(manifest_path: str = "outputs/metrics/dataset_splits.json") -> Dict[str, Any]:
    """Runs evaluation for both models and produces side-by-side comparison"""
    device = torch.device("cpu")
    threads = min(16, os.cpu_count() or 4)
    torch.set_num_threads(threads)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    loaders = create_dataloaders(
        train_records=manifest["train_records"],
        val_records=manifest["val_records"],
        test_records=manifest["test_records"],
        batch_size=32
    )

    models_to_eval = [
        ("custom_cnn", "checkpoints/best_driveguard_cnn.pt"),
        ("mobilenet_v2", "checkpoints/best_mobilenet_v2.pt")
    ]

    all_eval_results = {}
    comparison_table = []

    for model_type, ckpt_path in models_to_eval:
        print(f"\nEvaluating {model_type} on held-out test set (1,368 images)...")
        res = evaluate_single_model(model_type, ckpt_path, loaders["test"], device)
        all_eval_results[model_type] = res
        
        comparison_table.append({
            "Architecture": "DriveGuardCNN (From Scratch)" if model_type == "custom_cnn" else "DriveGuardMobileNetV2 (Transfer Learning)",
            "Test Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Balanced Accuracy (%)": round(res["balanced_accuracy"] * 100, 2),
            "Fatigue Recall (%)": round(res["per_class"]["Fatigue Subjects"]["recall"] * 100, 2),
            "Fatigue Precision (%)": round(res["per_class"]["Fatigue Subjects"]["precision"] * 100, 2),
            "Macro F1-Score (%)": round(res["macro_f1"] * 100, 2),
            "ROC-AUC": round(res["roc_auc"], 4),
            "Latency (ms/img)": res["avg_inference_latency_ms"],
            "Parameters": f"{res['trainable_parameters']:,}",
            "Model Size (MB)": res["checkpoint_size_mb"]
        })

    # Save detailed evaluation
    eval_path = Path("outputs/metrics/evaluation_results.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        # Exclude huge raw lists for compact file, save full details
        save_copy = {}
        for k, v in all_eval_results.items():
            save_copy[k] = {ik: iv for ik, iv in v.items() if ik != "test_predictions"}
            save_copy[k]["test_predictions_summary"] = {
                "total": len(v["test_predictions"]["targets"]),
                "targets_sample": v["test_predictions"]["targets"][:20],
                "preds_sample": v["test_predictions"]["preds"][:20],
                "probs_sample": v["test_predictions"]["probs"][:20]
            }
        json.dump(save_copy, f, indent=2)

    # Save full test predictions for ROC and error visualization
    predictions_path = Path("outputs/predictions/test_set_predictions.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w") as f:
        json.dump({
            "custom_cnn": all_eval_results["custom_cnn"]["test_predictions"],
            "mobilenet_v2": all_eval_results["mobilenet_v2"]["test_predictions"]
        }, f)

    # Save comparison table
    comp_path = Path("outputs/metrics/model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_table, f, indent=2)

    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET EVALUATION COMPARISON TABLE")
    print("=" * 80)
    for row in comparison_table:
        print(f"\nArchitecture: {row['Architecture']}")
        print(f"  Test Accuracy:        {row['Test Accuracy (%)']}%")
        print(f"  Balanced Accuracy:    {row['Balanced Accuracy (%)']}%")
        print(f"  Fatigue Recall:       {row['Fatigue Recall (%)']}% (Critical Safety Metric)")
        print(f"  Fatigue Precision:    {row['Fatigue Precision (%)']}%")
        print(f"  Macro F1-Score:       {row['Macro F1-Score (%)']}%")
        print(f"  ROC-AUC:              {row['ROC-AUC']}")
        print(f"  Inference Latency:    {row['Latency (ms/img)']} ms")
        print(f"  Parameters:           {row['Parameters']}")
        print(f"  Model Size:           {row['Model Size (MB)']} MB")

    return all_eval_results, comparison_table


if __name__ == "__main__":
    run_full_evaluation()
