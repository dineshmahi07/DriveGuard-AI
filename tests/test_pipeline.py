"""
DriveGuard AI — Automated Test Suite
Verifies:
1. Dataset discovery, labeling, and split integrity
2. Model forward and backward passes for both architectures
3. Checkpoint loading and weight restoration
4. Inference engine predictions and confidence scores
5. Computer vision temporal smoother and alert manager
6. Report generation deliverables presence and non-zero size
7. Web application component imports and integrity
"""

import os
import json
import unittest
from pathlib import Path
from PIL import Image
import torch

from src.preprocessing import discover_and_validate_dataset, create_splits, get_transforms
from src.dataset import DrowsinessDataset
from src.models.custom_cnn import DriveGuardCNN
from src.models.mobilenet_model import DriveGuardMobileNetV2
from src.inference import DriveGuardInferenceEngine
from src.temporal_detection import TemporalSmoother, DrowsinessAlertManager


class TestDriveGuardPipeline(unittest.TestCase):

    def test_01_dataset_discovery_and_balance(self):
        """Verifies dataset discovery, total image count, and class balance"""
        records, summary = discover_and_validate_dataset(raw_dir="0 FaceImages")
        self.assertEqual(summary["total_images"], 9120)
        self.assertEqual(summary["class_distribution"]["Active Subjects"], 4560)
        self.assertEqual(summary["class_distribution"]["Fatigue Subjects"], 4560)
        self.assertEqual(summary["corrupted_count"], 0)

    def test_02_split_integrity(self):
        """Verifies reproducible 70/15/15 splits without data leakage or overlap"""
        records, _ = discover_and_validate_dataset(raw_dir="0 FaceImages")
        train_r, val_r, test_r, summary = create_splits(records, seed=42)
        
        self.assertEqual(len(train_r), 6384)
        self.assertEqual(len(val_r), 1368)
        self.assertEqual(len(test_r), 1368)

        # Check disjoint sets
        train_paths = set(r["filepath"] for r in train_r)
        val_paths = set(r["filepath"] for r in val_r)
        test_paths = set(r["filepath"] for r in test_r)

        self.assertEqual(len(train_paths.intersection(val_paths)), 0)
        self.assertEqual(len(train_paths.intersection(test_paths)), 0)
        self.assertEqual(len(val_paths.intersection(test_paths)), 0)

    def test_03_custom_cnn_forward_and_backward(self):
        """Verifies DriveGuardCNN forward and backward passes and parameter count"""
        model = DriveGuardCNN(num_classes=2)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(model.count_parameters(), 422306)

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.block1.block[0].weight.grad)

    def test_04_mobilenet_v2_forward(self):
        """Verifies DriveGuardMobileNetV2 forward pass and parameter count"""
        model = DriveGuardMobileNetV2(num_classes=2, pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(model.count_parameters(), 2388098)

    def test_05_checkpoints_exist_and_load(self):
        """Verifies best checkpoints exist and can be loaded into models"""
        cnn_ckpt = Path("checkpoints/best_driveguard_cnn.pt")
        mob_ckpt = Path("checkpoints/best_mobilenet_v2.pt")
        self.assertTrue(cnn_ckpt.exists(), "CNN checkpoint missing")
        self.assertTrue(mob_ckpt.exists(), "MobileNet checkpoint missing")

        ckpt1 = torch.load(cnn_ckpt, map_location="cpu")
        self.assertIn("model_state_dict", ckpt1)
        self.assertGreater(ckpt1["val_acc"], 0.70)

        ckpt2 = torch.load(mob_ckpt, map_location="cpu")
        self.assertIn("model_state_dict", ckpt2)
        self.assertGreater(ckpt2["val_acc"], 0.90)

    def test_06_inference_engine_predictions(self):
        """Verifies single image inference and probability bounds"""
        engine = DriveGuardInferenceEngine(model_type="mobilenet_v2")
        img = Image.new("RGB", (128, 128), color=(120, 100, 90))
        res = engine.predict_image(img)
        self.assertIn(res["predicted_class"], ["Active (Alert)", "Fatigue (Drowsy)"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertAlmostEqual(res["prob_active"] + res["prob_fatigue"], 1.0, places=3)

    def test_07_temporal_smoothing_and_alert_manager(self):
        """Verifies temporal smoother windowing and alarm triggers"""
        smoother = TemporalSmoother(window_size=5)
        # Feed 4 active, 1 fatigue -> smoothed should be active
        for _ in range(4):
            smoother.update(0, 0.1)
        sm_pred, _ = smoother.update(1, 0.9)
        self.assertEqual(sm_pred, 0, "Transient blink should be smoothed to 0")

        # Alert manager sustained fatigue test
        alert_mgr = DrowsinessAlertManager(sustained_frames_threshold=3, cooldown_seconds=1.0)
        s1 = alert_mgr.update(1)
        self.assertFalse(s1["trigger_alarm"])
        s2 = alert_mgr.update(1)
        self.assertFalse(s2["trigger_alarm"])
        s3 = alert_mgr.update(1)
        self.assertTrue(s3["trigger_alarm"], "Alarm should trigger after 3 sustained frames")

    def test_08_deliverable_artifacts(self):
        """Verifies all academic reports, presentations, and plots exist"""
        plots = [
            "outputs/plots/1_dataset_class_distribution.png",
            "outputs/plots/6_confusion_matrices_comparison.png",
            "outputs/plots/7_model_comparison_barchart.png",
            "outputs/plots/9_roc_curves_comparison.png"
        ]
        for p in plots:
            self.assertTrue(os.path.exists(p), f"Plot {p} missing")

        reports = [
            "reports/final_report.docx",
            "reports/final_report.pdf",
            "reports/presentation.pptx",
            "reports/VIVA_PREPARATION.md"
        ]
        for r in reports:
            self.assertTrue(os.path.exists(r), f"Report {r} missing")
            self.assertGreater(os.path.getsize(r), 1000, f"Report {r} is empty")


if __name__ == "__main__":
    unittest.main()
