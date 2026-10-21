"""
DriveGuard AI — Computer Vision Techniques & Real-Time Temporal Stabilization
Implements:
1. Temporal Moving Average & Rolling Window Smoothing for Video Streams
2. Persistent Drowsiness Duration Tracking (State Machine)
3. Confidence Threshold Optimization (Safety-Critical Recall Tuning)
4. Model Ensembling (Soft Voting)
5. Alert Cooldown Management
"""

import os
import json
import time
from collections import deque
from typing import List, Dict, Any, Tuple
import numpy as np


class TemporalSmoother:
    """
    Maintains a rolling FIFO window of recent frame predictions and probabilities
    to filter out transient blinks, head turns, and sensor noise.
    """
    def __init__(self, window_size: int = 10, alpha: float = 0.35):
        self.window_size = window_size
        self.alpha = alpha  # EMA smoothing factor
        self.pred_history = deque(maxlen=window_size)
        self.prob_history = deque(maxlen=window_size)
        self.ema_prob = None

    def update(self, raw_pred: int, raw_prob_fatigue: float) -> Tuple[int, float]:
        """
        Updates buffer with current frame inference and returns smoothed (pred, prob).
        """
        self.pred_history.append(raw_pred)
        self.prob_history.append(raw_prob_fatigue)

        # Exponential Moving Average
        if self.ema_prob is None:
            self.ema_prob = raw_prob_fatigue
        else:
            self.ema_prob = self.alpha * raw_prob_fatigue + (1.0 - self.alpha) * self.ema_prob

        # Rolling majority voting
        majority_pred = 1 if sum(self.pred_history) > (len(self.pred_history) / 2.0) else 0

        return majority_pred, float(self.ema_prob)

    def reset(self):
        self.pred_history.clear()
        self.prob_history.clear()
        self.ema_prob = None


class DrowsinessAlertManager:
    """
    Monitors duration of sustained drowsiness, triggers audio/visual alarms,
    and manages alarm cooldown periods.
    """
    def __init__(self, sustained_frames_threshold: int = 5, cooldown_seconds: float = 3.0):
        self.sustained_threshold = sustained_frames_threshold
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_fatigue_frames = 0
        self.last_alarm_time = 0.0
        self.is_alarm_active = False

    def update(self, smoothed_pred: int, current_timestamp: float = None) -> Dict[str, Any]:
        if current_timestamp is None:
            current_timestamp = time.time()

        if smoothed_pred == 1:
            self.consecutive_fatigue_frames += 1
        else:
            self.consecutive_fatigue_frames = max(0, self.consecutive_fatigue_frames - 1)

        time_since_alarm = current_timestamp - self.last_alarm_time
        trigger_alarm = False

        if self.consecutive_fatigue_frames >= self.sustained_threshold:
            if time_since_alarm >= self.cooldown_seconds:
                trigger_alarm = True
                self.last_alarm_time = current_timestamp
                self.is_alarm_active = True
        else:
            self.is_alarm_active = False

        return {
            "consecutive_fatigue_frames": self.consecutive_fatigue_frames,
            "threshold_required": self.sustained_threshold,
            "trigger_alarm": trigger_alarm,
            "is_alarm_active": self.is_alarm_active,
            "seconds_since_last_alarm": round(time_since_alarm, 2)
        }

    def reset(self):
        self.consecutive_fatigue_frames = 0
        self.last_alarm_time = 0.0
        self.is_alarm_active = False


def evaluate_confidence_thresholds(
    y_true: np.ndarray,
    y_prob_fatigue: np.ndarray,
    thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
) -> List[Dict[str, Any]]:
    """
    Analyzes tradeoff between False Positives (annoyance) and False Negatives (danger)
    across different classification decision thresholds.
    """
    analysis = []
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    for th in thresholds:
        preds = (y_prob_fatigue >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true)

        analysis.append({
            "threshold": round(th, 2),
            "fatigue_recall": round(recall * 100, 2),
            "fatigue_precision": round(precision * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "accuracy": round(accuracy * 100, 2),
            "false_positives": fp,
            "false_negatives": fn
        })
    return analysis


def evaluate_ensemble(
    y_true: np.ndarray,
    probs_cnn: np.ndarray,
    probs_mobilenet: np.ndarray,
    weight_mobilenet: float = 0.70
) -> Dict[str, Any]:
    """
    Evaluates soft-voting ensemble combining DriveGuardCNN and DriveGuardMobileNetV2.
    """
    ensemble_probs = weight_mobilenet * probs_mobilenet + (1.0 - weight_mobilenet) * probs_cnn
    preds = (ensemble_probs >= 0.5).astype(int)
    
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))

    acc = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "weight_mobilenet": weight_mobilenet,
        "weight_cnn": round(1.0 - weight_mobilenet, 2),
        "accuracy": round(acc * 100, 2),
        "fatigue_recall": round(recall * 100, 2),
        "fatigue_precision": round(precision * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "false_positives": fp,
        "false_negatives": fn,
        "ensemble_probs": ensemble_probs.tolist()
    }


def run_cv_techniques_evaluation():
    """
    Loads test predictions and runs quantitative evaluation of:
    - Confidence threshold optimization
    - Model ensembling
    - Temporal smoothing simulation
    """
    pred_path = "outputs/predictions/test_set_predictions.json"
    with open(pred_path, "r") as f:
        data = json.load(f)

    y_true = np.array(data["mobilenet_v2"]["targets"])
    probs_cnn = np.array(data["custom_cnn"]["probs"])
    probs_mob = np.array(data["mobilenet_v2"]["probs"])

    print("Running Confidence Threshold Analysis for MobileNetV2...")
    threshold_results = evaluate_confidence_thresholds(y_true, probs_mob)

    print("Running Ensemble Analysis (MobileNetV2 70% + CNN 30%)...")
    ensemble_results = evaluate_ensemble(y_true, probs_cnn, probs_mob, weight_mobilenet=0.75)

    # Simulate temporal smoothing on a 50-frame sequence
    smoother = TemporalSmoother(window_size=7, alpha=0.3)
    # Generate a realistic synthetic sequence with transient blinks/noise
    seq_true = [0]*15 + [1]*25 + [0]*10
    noisy_preds = list(seq_true)
    # Add 2 transient false positives (spurious blink while active)
    noisy_preds[5] = 1
    noisy_preds[7] = 1
    # Add 2 transient false negatives (head tilt during drowsiness)
    noisy_preds[25] = 0
    noisy_preds[28] = 0

    smoothed_preds = []
    smoothed_probs = []
    for raw_p in noisy_preds:
        raw_prob = 0.85 if raw_p == 1 else 0.15
        sp, sprob = smoother.update(raw_p, raw_prob)
        smoothed_preds.append(sp)
        smoothed_probs.append(round(sprob, 3))

    # Calculate reduction in state oscillations (chatter)
    raw_flips = sum(1 for i in range(1, len(noisy_preds)) if noisy_preds[i] != noisy_preds[i-1])
    smoothed_flips = sum(1 for i in range(1, len(smoothed_preds)) if smoothed_preds[i] != smoothed_preds[i-1])
    noise_reduction_pct = round(((raw_flips - smoothed_flips) / raw_flips) * 100, 2)

    cv_report = {
        "confidence_threshold_analysis": threshold_results,
        "ensemble_results": {k: v for k, v in ensemble_results.items() if k != "ensemble_probs"},
        "temporal_smoothing_experiment": {
            "window_size": 7,
            "ema_alpha": 0.3,
            "raw_transitions_flicker": raw_flips,
            "smoothed_transitions_flicker": smoothed_flips,
            "oscillation_reduction_pct": noise_reduction_pct,
            "simulation_sequence_length": len(noisy_preds),
            "sim_ground_truth": seq_true,
            "sim_noisy_raw": noisy_preds,
            "sim_smoothed": smoothed_preds
        }
    }

    out_file = "outputs/metrics/cv_techniques_experiments.json"
    with open(out_file, "w") as f:
        json.dump(cv_report, f, indent=2)

    print(f"Computer Vision techniques report saved to {out_file}")
    print(f"  Ensemble Accuracy: {ensemble_results['accuracy']}% (Recall: {ensemble_results['fatigue_recall']}%)")
    print(f"  Temporal Smoothing Reduced State Oscillations by {noise_reduction_pct}%")


if __name__ == "__main__":
    run_cv_techniques_evaluation()
