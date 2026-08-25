"""
Module M5: Drift Detection & Monitoring
Detects distribution shift in model predictions and input data.

Drift Detection Strategy (Justification):
1. Confidence Score Monitoring — Track rolling mean confidence. A sustained drop
   indicates the model is encountering unfamiliar inputs (e.g., new lighting conditions).
2. Prediction Distribution Shift — Use KS-test to compare recent prediction 
   distributions against a reference (training) distribution.
3. Input Feature Drift — Monitor image statistics (brightness, contrast) for 
   changes that indicate environmental shifts.

These three signals together provide robust drift detection:
- Confidence alone might miss cases where the model is confidently wrong
- Distribution shift catches class-ratio changes
- Input feature monitoring catches upstream data quality issues
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Multi-signal drift detection for deployed defect classifier.
    
    Monitors:
    1. Prediction confidence degradation
    2. Class distribution shift (KS-test)
    3. Input image statistics shift
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        ks_pvalue_threshold: float = 0.05,
        window_size: int = 100,
        reference_confidence: Optional[List[float]] = None,
    ):
        """
        Args:
            confidence_threshold: Alert if rolling mean confidence drops below this
            ks_pvalue_threshold: Alert if KS-test p-value below this (significant shift)
            window_size: Number of recent predictions to consider
            reference_confidence: Baseline confidence scores from validation set
        """
        self.confidence_threshold = confidence_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.window_size = window_size
        self.reference_confidence = reference_confidence or []

        # Storage
        self.prediction_history: List[Dict] = []
        self.drift_alerts: List[Dict] = []
        self.monitoring_log: List[Dict] = []

    def log_prediction(self, prediction: Dict) -> Optional[Dict]:
        """
        Log a prediction and check for drift.
        
        Args:
            prediction: Dict with keys: confidence, label, timestamp, 
                       image_brightness (optional), image_contrast (optional)
                       
        Returns:
            Alert dict if drift detected, None otherwise
        """
        self.prediction_history.append(prediction)

        # Only check drift when we have enough data
        if len(self.prediction_history) < self.window_size:
            return None

        # Run drift checks
        alert = self._check_all_signals()
        if alert:
            self.drift_alerts.append(alert)
            logger.warning(f"DRIFT ALERT: {alert}")

        return alert

    def _check_all_signals(self) -> Optional[Dict]:
        """Run all drift detection checks."""
        recent = self.prediction_history[-self.window_size:]
        alerts = []

        # Signal 1: Confidence degradation
        confidence_alert = self._check_confidence_drift(recent)
        if confidence_alert:
            alerts.append(confidence_alert)

        # Signal 2: Prediction distribution shift
        if self.reference_confidence:
            dist_alert = self._check_distribution_drift(recent)
            if dist_alert:
                alerts.append(dist_alert)

        # Signal 3: Input feature drift
        feature_alert = self._check_input_drift(recent)
        if feature_alert:
            alerts.append(feature_alert)

        if alerts:
            combined_alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "signals": alerts,
                "severity": "high" if len(alerts) >= 2 else "medium",
                "recommendation": self._get_recommendation(alerts),
            }
            self.monitoring_log.append(combined_alert)
            return combined_alert

        # Log healthy status
        self.monitoring_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "rolling_confidence": np.mean([p["confidence"] for p in recent]),
        })
        return None

    def _check_confidence_drift(self, recent: List[Dict]) -> Optional[Dict]:
        """Check if confidence scores have degraded."""
        confidences = [p["confidence"] for p in recent]
        rolling_mean = np.mean(confidences)
        rolling_std = np.std(confidences)

        if rolling_mean < self.confidence_threshold:
            return {
                "signal": "confidence_degradation",
                "rolling_mean": float(rolling_mean),
                "rolling_std": float(rolling_std),
                "threshold": self.confidence_threshold,
                "message": f"Rolling confidence ({rolling_mean:.3f}) below "
                           f"threshold ({self.confidence_threshold})",
            }
        return None

    def _check_distribution_drift(self, recent: List[Dict]) -> Optional[Dict]:
        """Use KS-test to detect shift in confidence distribution."""
        recent_confidences = [p["confidence"] for p in recent]

        # KS-test: compare recent vs reference
        ks_stat, p_value = stats.ks_2samp(
            self.reference_confidence, recent_confidences
        )

        if p_value < self.ks_pvalue_threshold:
            return {
                "signal": "distribution_shift",
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "threshold": self.ks_pvalue_threshold,
                "message": f"Significant distribution shift detected "
                           f"(KS={ks_stat:.4f}, p={p_value:.4f})",
            }
        return None

    def _check_input_drift(self, recent: List[Dict]) -> Optional[Dict]:
        """Check for drift in input image statistics."""
        # Only if image stats are logged
        brightness_values = [
            p.get("image_brightness") for p in recent
            if p.get("image_brightness") is not None
        ]

        if not brightness_values:
            return None

        mean_brightness = np.mean(brightness_values)

        # Alert if brightness significantly different from expected (0.4-0.6 range)
        if mean_brightness < 0.3 or mean_brightness > 0.7:
            return {
                "signal": "input_feature_drift",
                "mean_brightness": float(mean_brightness),
                "message": f"Input brightness drift detected "
                           f"(mean={mean_brightness:.3f}, expected 0.4-0.6)",
            }
        return None

    def _get_recommendation(self, alerts: List[Dict]) -> str:
        """Generate actionable recommendation based on alerts."""
        signals = [a["signal"] for a in alerts]

        if "confidence_degradation" in signals and "distribution_shift" in signals:
            return ("RETRAIN RECOMMENDED: Both confidence and distribution metrics "
                    "indicate model performance has degraded. Collect recent samples "
                    "and trigger retraining pipeline.")
        elif "input_feature_drift" in signals:
            return ("INVESTIGATE INPUT PIPELINE: Input data characteristics have "
                    "changed. Check camera settings, lighting conditions, or "
                    "upstream preprocessing.")
        elif "confidence_degradation" in signals:
            return ("MONITOR CLOSELY: Confidence is dropping. If trend continues "
                    "for 2+ windows, trigger retraining.")
        else:
            return "Continue monitoring. Single-signal alert may be transient."

    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.prediction_history:
            return {"status": "no_data"}

        recent = self.prediction_history[-self.window_size:]
        confidences = [p["confidence"] for p in recent]

        return {
            "total_predictions": len(self.prediction_history),
            "total_alerts": len(self.drift_alerts),
            "rolling_confidence_mean": float(np.mean(confidences)),
            "rolling_confidence_std": float(np.std(confidences)),
            "recent_defective_ratio": sum(
                1 for p in recent if p.get("label") == "defective"
            ) / len(recent),
            "last_alert": self.drift_alerts[-1] if self.drift_alerts else None,
        }

    def save_monitoring_log(self, output_path: str = "logs/monitoring_log.json"):
        """Save monitoring log to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({
                "monitoring_log": self.monitoring_log,
                "drift_alerts": self.drift_alerts,
                "summary": self.get_summary(),
            }, f, indent=2, default=str)
        logger.info(f"Monitoring log saved to {output_path}")
