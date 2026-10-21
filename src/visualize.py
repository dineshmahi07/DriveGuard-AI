"""
DriveGuard AI — Academic Visualization Suite
Generates 12 high-resolution publication-quality figures based exclusively on
real experimental results and saved evaluation metrics.
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_1_class_distribution():
    """1. Dataset class distribution bar chart"""
    with open("outputs/metrics/dataset_splits.json", "r") as f:
        data = json.load(f)["split_summary"]

    splits = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    active = [data["train"]["active"], data["val"]["active"], data["test"]["active"]]
    fatigue = [data["train"]["fatigue"], data["val"]["fatigue"], data["test"]["fatigue"]]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, active, width, label='Active Subjects', color='#1f77b4', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x + width/2, fatigue, width, label='Fatigue Subjects', color='#d62728', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Number of Images')
    ax.set_title('DriveGuard AI: Dataset Class Distribution Across Reproducible Splits', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend(frameon=True)
    ax.set_ylim(0, 3600)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_dataset_class_distribution.png", dpi=300)
    plt.close()
    print("Generated 1_dataset_class_distribution.png")


def plot_2_representative_samples():
    """2. Representative samples from both classes"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    classes = ["Active Subjects", "Fatigue Subjects"]
    titles = ["Attentive Looking", "Alert Driving", "Attentive Facing", "Active Monitoring",
              "Severe Yawn", "Drowsy Nod", "Eyes Drooping", "Yawning Driver"]

    for row, cls in enumerate(classes):
        cls_dir = os.path.join("0 FaceImages", cls)
        files = os.listdir(cls_dir)[:4]
        for col, f in enumerate(files):
            img_path = os.path.join(cls_dir, f)
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{cls}\n{titles[row*4 + col]} ({img.size[0]}x{img.size[1]})", fontsize=9, fontweight='bold')
            axes[row, col].axis('off')

    plt.suptitle("DriveGuard AI: Representative Dataset Samples (Active vs Fatigue Subjects)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_dataset_representative_samples.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 2_dataset_representative_samples.png")


def plot_3_preprocessing_augmentation():
    """3. Preprocessing and data augmentation demonstrations"""
    from torchvision import transforms

    raw_path = os.path.join("0 FaceImages", "Active Subjects", os.listdir("0 FaceImages/Active Subjects")[0])
    img = Image.open(raw_path)

    augment_transforms = [
        ("Original Raw Image", None),
        ("Standard Resize (128x128)", transforms.Resize((128, 128))),
        ("Random Horizontal Flip", transforms.Compose([transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(p=1.0)])),
        ("Rotation (10 deg)", transforms.Compose([transforms.Resize((128, 128)), transforms.RandomRotation(degrees=(10, 10))])),
        ("Color Jitter (Brightness/Contrast)", transforms.Compose([transforms.Resize((128, 128)), transforms.ColorJitter(brightness=0.3, contrast=0.3)]))
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for idx, (title, tr) in enumerate(augment_transforms):
        aug_img = tr(img) if tr is not None else img
        axes[idx].imshow(aug_img)
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle("DriveGuard AI: Preprocessing and Data Augmentation Pipeline", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_preprocessing_augmentation_samples.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 3_preprocessing_augmentation_samples.png")


def plot_4_accuracy_curves():
    """4. Training vs Validation accuracy curves for both models"""
    with open("outputs/metrics/history_custom_cnn.json", "r") as f:
        h_cnn = json.load(f)
    with open("outputs/metrics/history_mobilenet_v2.json", "r") as f:
        h_mob = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CNN curves
    epochs_cnn = list(range(1, len(h_cnn["train_acc"]) + 1))
    ax1.plot(epochs_cnn, [a * 100 for a in h_cnn["train_acc"]], 'o-', label='Train Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_cnn, [a * 100 for a in h_cnn["val_acc"]], 's--', label='Val Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title(f'DriveGuardCNN (Trained from Scratch)\nBest Val Accuracy: {h_cnn["val_acc"][h_cnn["best_epoch"]-1]*100:.2f}%', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(50, 100)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # MobileNet curves
    epochs_mob = list(range(1, len(h_mob["train_acc"]) + 1))
    ax2.plot(epochs_mob, [a * 100 for a in h_mob["train_acc"]], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_mob, [a * 100 for a in h_mob["val_acc"]], 's--', label='Val Accuracy', color='#d62728', linewidth=2)
    ax2.set_title(f'DriveGuardMobileNetV2 (Transfer Learning)\nBest Val Accuracy: {h_mob["val_acc"][h_mob["best_epoch"]-1]*100:.2f}%', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(50, 100)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training vs. Validation Accuracy Trajectories", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_training_validation_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 4_training_validation_accuracy.png")


def plot_5_loss_curves():
    """5. Training vs Validation loss curves for both models"""
    with open("outputs/metrics/history_custom_cnn.json", "r") as f:
        h_cnn = json.load(f)
    with open("outputs/metrics/history_mobilenet_v2.json", "r") as f:
        h_mob = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_cnn = list(range(1, len(h_cnn["train_loss"]) + 1))
    ax1.plot(epochs_cnn, h_cnn["train_loss"], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_cnn, h_cnn["val_loss"], 's--', label='Val Loss', color='#ff7f0e', linewidth=2)
    ax1.set_title('DriveGuardCNN Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    epochs_mob = list(range(1, len(h_mob["train_loss"]) + 1))
    ax2.plot(epochs_mob, h_mob["train_loss"], 'o-', label='Train Loss', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_mob, h_mob["val_loss"], 's--', label='Val Loss', color='#d62728', linewidth=2)
    ax2.set_title('DriveGuardMobileNetV2 Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training vs. Validation Cross-Entropy Loss Curves", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_training_validation_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 5_training_validation_loss.png")


def plot_6_confusion_matrices():
    """6. Confusion matrix heatmaps for both models"""
    with open("outputs/metrics/evaluation_results.json", "r") as f:
        eval_data = json.load(f)

    cm_cnn = np.array(eval_data["custom_cnn"]["confusion_matrix"])
    cm_mob = np.array(eval_data["mobilenet_v2"]["confusion_matrix"])
    labels = ["Active", "Fatigue"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title(f"DriveGuardCNN (From Scratch)\nTest Acc: {eval_data['custom_cnn']['accuracy']*100:.1f}%, Fatigue Recall: {eval_data['custom_cnn']['per_class']['Fatigue Subjects']['recall']*100:.1f}%", fontweight='bold')
    ax1.set_xlabel("Predicted Driver State")
    ax1.set_ylabel("True Driver State")

    sns.heatmap(cm_mob, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax2.set_title(f"DriveGuardMobileNetV2 (Transfer Learning)\nTest Acc: {eval_data['mobilenet_v2']['accuracy']*100:.1f}%, Fatigue Recall: {eval_data['mobilenet_v2']['per_class']['Fatigue Subjects']['recall']*100:.1f}%", fontweight='bold')
    ax2.set_xlabel("Predicted Driver State")
    ax2.set_ylabel("True Driver State")

    plt.suptitle("Confusion Matrix Heatmaps on Held-Out Test Set (N = 1,368)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "6_confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 6_confusion_matrices_comparison.png")


def plot_7_model_comparison_bars():
    """7. Side-by-side metric comparison bar chart"""
    with open("outputs/metrics/evaluation_results.json", "r") as f:
        ev = json.load(f)

    metrics = ["Accuracy", "Balanced Acc", "Fatigue Recall", "Fatigue Precision", "Macro F1", "ROC-AUC"]
    cnn_vals = [
        ev["custom_cnn"]["accuracy"] * 100,
        ev["custom_cnn"]["balanced_accuracy"] * 100,
        ev["custom_cnn"]["per_class"]["Fatigue Subjects"]["recall"] * 100,
        ev["custom_cnn"]["per_class"]["Fatigue Subjects"]["precision"] * 100,
        ev["custom_cnn"]["macro_f1"] * 100,
        ev["custom_cnn"]["roc_auc"] * 100
    ]
    mob_vals = [
        ev["mobilenet_v2"]["accuracy"] * 100,
        ev["mobilenet_v2"]["balanced_accuracy"] * 100,
        ev["mobilenet_v2"]["per_class"]["Fatigue Subjects"]["recall"] * 100,
        ev["mobilenet_v2"]["per_class"]["Fatigue Subjects"]["precision"] * 100,
        ev["mobilenet_v2"]["macro_f1"] * 100,
        ev["mobilenet_v2"]["roc_auc"] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width/2, cnn_vals, width, label='DriveGuardCNN (Scratch)', color='#4a90e2', edgecolor='black')
    r2 = ax.bar(x + width/2, mob_vals, width, label='DriveGuardMobileNetV2 (Transfer)', color='#50e3c2', edgecolor='black')

    ax.set_ylabel('Score (%)')
    ax.set_title('Comprehensive Architecture Performance Comparison on Held-Out Test Data', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc='upper left')

    for rect in r1 + r2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "7_model_comparison_barchart.png", dpi=300)
    plt.close()
    print("Generated 7_model_comparison_barchart.png")


def plot_8_per_class_metrics():
    """8. Per-class precision, recall, and F1-score chart"""
    with open("outputs/metrics/evaluation_results.json", "r") as f:
        ev = json.load(f)

    categories = ["Active Precision", "Active Recall", "Active F1", "Fatigue Precision", "Fatigue Recall", "Fatigue F1"]
    cnn_p = ev["custom_cnn"]["per_class"]
    mob_p = ev["mobilenet_v2"]["per_class"]

    cnn_scores = [
        cnn_p["Active Subjects"]["precision"] * 100,
        cnn_p["Active Subjects"]["recall"] * 100,
        cnn_p["Active Subjects"]["f1_score"] * 100,
        cnn_p["Fatigue Subjects"]["precision"] * 100,
        cnn_p["Fatigue Subjects"]["recall"] * 100,
        cnn_p["Fatigue Subjects"]["f1_score"] * 100
    ]
    mob_scores = [
        mob_p["Active Subjects"]["precision"] * 100,
        mob_p["Active Subjects"]["recall"] * 100,
        mob_p["Active Subjects"]["f1_score"] * 100,
        mob_p["Fatigue Subjects"]["precision"] * 100,
        mob_p["Fatigue Subjects"]["recall"] * 100,
        mob_p["Fatigue Subjects"]["f1_score"] * 100
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width/2, cnn_scores, width, label='DriveGuardCNN', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, mob_scores, width, label='DriveGuardMobileNetV2', color='#2ca02c', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Metric Score (%)')
    ax.set_title('Per-Class Diagnostic Metrics (Active vs Fatigue Classification)', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(frameon=True)

    for i in range(len(categories)):
        ax.text(x[i] - width/2, cnn_scores[i] + 1.5, f"{cnn_scores[i]:.1f}%", ha='center', fontsize=8)
        ax.text(x[i] + width/2, mob_scores[i] + 1.5, f"{mob_scores[i]:.1f}%", ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "8_per_class_metrics_comparison.png", dpi=300)
    plt.close()
    print("Generated 8_per_class_metrics_comparison.png")


def plot_9_roc_curves():
    """9. ROC curves for both models with AUC scores"""
    with open("outputs/predictions/test_set_predictions.json", "r") as f:
        preds = json.load(f)

    y_true = np.array(preds["mobilenet_v2"]["targets"])
    probs_cnn = np.array(preds["custom_cnn"]["probs"])
    probs_mob = np.array(preds["mobilenet_v2"]["probs"])

    fpr_cnn, tpr_cnn, _ = roc_curve(y_true, probs_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    fpr_mob, tpr_mob, _ = roc_curve(y_true, probs_mob)
    auc_mob = auc(fpr_mob, tpr_mob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_cnn, tpr_cnn, color='#1f77b4', lw=2.5, label=f'DriveGuardCNN (AUC = {auc_cnn:.4f})')
    plt.plot(fpr_mob, tpr_mob, color='#2ca02c', lw=2.5, label=f'DriveGuardMobileNetV2 (AUC = {auc_mob:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Active Misclassified as Fatigue)')
    plt.ylabel('True Positive Rate (Sensitivity / Fatigue Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', pad=15, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "9_roc_curves_comparison.png", dpi=300)
    plt.close()
    print("Generated 9_roc_curves_comparison.png")


def plot_10_error_analysis():
    """10. Error analysis visual displaying representative incorrect predictions"""
    with open("outputs/metrics/evaluation_results.json", "r") as f:
        ev = json.load(f)

    misclassified = ev["mobilenet_v2"]["sample_misclassified"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, sample in enumerate(misclassified):
        fname = sample["filename"]
        # Look up true path
        folder = "Fatigue Subjects" if sample["true_label"] == 1 else "Active Subjects"
        img_path = os.path.join("0 FaceImages", folder, fname)
        
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
        except Exception:
            axes[idx].text(0.5, 0.5, "Image Unreadable", ha='center', va='center')

        true_txt = "Fatigue" if sample["true_label"] == 1 else "Active"
        pred_txt = "Fatigue" if sample["predicted_label"] == 1 else "Active"
        prob = sample["predicted_prob_fatigue"]
        
        axes[idx].set_title(
            f"Filename: {fname}\n"
            f"True: {true_txt} | Predicted: {pred_txt}\n"
            f"Fatigue Confidence: {prob*100:.1f}%",
            fontsize=9, color='darkred', fontweight='bold'
        )
        axes[idx].axis('off')

    plt.suptitle("DriveGuard AI: Error Analysis — Representative Misclassified Test Cases", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_error_analysis_misclassifications.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 10_error_analysis_misclassifications.png")


def plot_11_confidence_distribution():
    """11. Prediction confidence distribution histograms"""
    with open("outputs/predictions/test_set_predictions.json", "r") as f:
        preds = json.load(f)

    probs_mob = np.array(preds["mobilenet_v2"]["probs"])
    targets = np.array(preds["mobilenet_v2"]["targets"])

    active_probs = probs_mob[targets == 0]
    fatigue_probs = probs_mob[targets == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(active_probs, bins=25, alpha=0.65, label='True Active Subjects', color='#1f77b4', edgecolor='black')
    ax.hist(fatigue_probs, bins=25, alpha=0.65, label='True Fatigue Subjects', color='#d62728', edgecolor='black')

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Standard Decision Threshold (0.50)')
    ax.set_xlabel('Predicted Probability of Fatigue P(Drowsy)')
    ax.set_ylabel('Number of Test Images')
    ax.set_title('DriveGuardMobileNetV2: Test Prediction Confidence Distribution', pad=15, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_confidence_distribution.png", dpi=300)
    plt.close()
    print("Generated 11_confidence_distribution.png")


def plot_12_technique_comparison():
    """12. Computer vision technique comparison chart (smoothing & thresholding)"""
    with open("outputs/metrics/cv_techniques_experiments.json", "r") as f:
        cv_exp = json.load(f)

    th_data = cv_exp["confidence_threshold_analysis"]
    thresholds = [d["threshold"] for d in th_data]
    recalls = [d["fatigue_recall"] for d in th_data]
    precisions = [d["fatigue_precision"] for d in th_data]
    f1s = [d["f1_score"] for d in th_data]
    fps = [d["false_positives"] for d in th_data]
    fns = [d["false_negatives"] for d in th_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Threshold vs Metrics
    ax1.plot(thresholds, recalls, 'o-', label='Fatigue Recall (%)', color='#d62728', lw=2)
    ax1.plot(thresholds, precisions, 's-', label='Fatigue Precision (%)', color='#1f77b4', lw=2)
    ax1.plot(thresholds, f1s, '^--', label='F1-Score (%)', color='#2ca02c', lw=2)
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Threshold Tuning: Safety Recall vs False Alarms', fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # FP vs FN tradeoff
    ax2.plot(thresholds, fps, 'x-', label='False Positives (Nuisance Alarms)', color='#ff7f0e', lw=2)
    ax2.plot(thresholds, fns, 'd-', label='False Negatives (Missed Drowsiness)', color='#d62728', lw=2)
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Count of Error Cases (out of 1,368)')
    ax2.set_title('False Positives vs. False Negatives Tradeoff', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("DriveGuard AI: Computer Vision Decision Optimization Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_technique_comparison_charts.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated 12_technique_comparison_charts.png")


def generate_all_plots():
    print("Starting generation of all 12 publication-quality visualizations...")
    plot_1_class_distribution()
    plot_2_representative_samples()
    plot_3_preprocessing_augmentation()
    plot_4_accuracy_curves()
    plot_5_loss_curves()
    plot_6_confusion_matrices()
    plot_7_model_comparison_bars()
    plot_8_per_class_metrics()
    plot_9_roc_curves()
    plot_10_error_analysis()
    plot_11_confidence_distribution()
    plot_12_technique_comparison()
    print("All 12 visualizations successfully generated in outputs/plots/!")


if __name__ == "__main__":
    generate_all_plots()
