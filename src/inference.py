"""
DriveGuard AI — Real-Time Inference Engine
Loads trained model checkpoints and runs preprocessing, forward inference,
confidence calibration, and computer vision temporal stabilization on single images
or streaming webcam video frames.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing import get_transforms
from src.models.custom_cnn import DriveGuardCNN
from src.models.mobilenet_model import DriveGuardMobileNetV2
from src.temporal_detection import TemporalSmoother, DrowsinessAlertManager


class DriveGuardInferenceEngine:
    """
    Production inference engine supporting both DriveGuardCNN and DriveGuardMobileNetV2.
    Integrates real-time preprocessing, confidence scoring, and temporal filtering.
    """
    def __init__(
        self,
        model_type: str = "mobilenet_v2",
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        temporal_window: int = 7,
        alert_sustained_frames: int = 5,
        alert_cooldown_sec: float = 3.0
    ):
        self.device = torch.device("cpu")
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.transform = get_transforms(image_size=(128, 128), augment=False)
        self.classes = ["Active", "Fatigue"]

        # Default checkpoint paths
        project_root = Path(__file__).resolve().parent.parent
        if checkpoint_path is None:
            if "mobilenet" in self.model_type:
                checkpoint_path = project_root / "checkpoints" / "best_mobilenet_v2.pt"
            else:
                checkpoint_path = project_root / "checkpoints" / "best_driveguard_cnn.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / checkpoint_path

        self.checkpoint_path = str(checkpoint_path)
        self._load_model()

        # CV temporal stabilization components
        self.smoother = TemporalSmoother(window_size=temporal_window, alpha=0.35)
        self.alert_manager = DrowsinessAlertManager(
            sustained_frames_threshold=alert_sustained_frames,
            cooldown_seconds=alert_cooldown_sec
        )

    def _load_model(self):
        """Loads weights from disk into evaluation mode"""
        if "mobilenet" in self.model_type:
            self.model = DriveGuardMobileNetV2(num_classes=2, pretrained=False)
        else:
            self.model = DriveGuardCNN(num_classes=2)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_metadata = {
            "epoch": ckpt.get("epoch", 0),
            "val_acc": ckpt.get("val_acc", 0.0),
            "val_f1": ckpt.get("val_f1", 0.0)
        }

    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs single image classification.
        Returns predicted state, confidence, and calibrated fatigue probability.
        """
        image_rgb = image.convert("RGB")
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        prob_active = float(probs[0])
        prob_fatigue = float(probs[1])

        # Apply configurable classification threshold
        is_fatigue = prob_fatigue >= self.confidence_threshold
        pred_label = 1 if is_fatigue else 0
        pred_class = "Fatigue (Drowsy)" if is_fatigue else "Active (Alert)"
        confidence = prob_fatigue if is_fatigue else prob_active

        return {
            "predicted_class": pred_class,
            "label_idx": pred_label,
            "confidence": round(confidence, 4),
            "prob_fatigue": round(prob_fatigue, 4),
            "prob_active": round(prob_active, 4),
            "threshold_used": self.confidence_threshold,
            "model_type": self.model_type
        }

    def predict_frame_stream(self, frame_image: Image.Image, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs temporal video stream inference with rolling smoothing and sustained alert tracking.
        """
        raw_res = self.predict_image(frame_image)
        smoothed_pred, smoothed_fatigue_prob = self.smoother.update(
            raw_res["label_idx"],
            raw_res["prob_fatigue"]
        )
        alert_status = self.alert_manager.update(smoothed_pred, timestamp)

        smoothed_class = "Fatigue (Drowsy)" if smoothed_pred == 1 else "Active (Alert)"

        return {
            "raw_prediction": raw_res,
            "smoothed_class": smoothed_class,
            "smoothed_label": smoothed_pred,
            "smoothed_fatigue_prob": round(smoothed_fatigue_prob, 4),
            "alert_status": alert_status
        }

    def reset_stream(self):
        """Resets temporal smoothing buffer and alert timer"""
        self.smoother.reset()
        self.alert_manager.reset()


if __name__ == "__main__":
    engine = DriveGuardInferenceEngine(model_type="mobilenet_v2")
    sample_path = "0 FaceImages/Active Subjects/" + os.listdir("0 FaceImages/Active Subjects")[0]
    sample_img = Image.open(sample_path)
    res = engine.predict_image(sample_img)
    print("Test inference result:", res)
