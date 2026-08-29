"""
Unit tests for drift detection and retraining logic (M5).
"""

import pytest
import numpy as np

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.retrain_strategy import RetrainingTrigger


class TestDriftDetector:
    def test_no_alert_with_high_confidence(self):
        """No drift alert when confidence matches reference distribution."""
        # Use reference that matches what we'll send (no distribution shift)
        reference = [0.90 + 0.05 * (i % 3) for i in range(100)]
        detector = DriftDetector(
            confidence_threshold=0.7,
            window_size=10,
            reference_confidence=reference
        )

        for i in range(20):
            alert = detector.log_prediction({
                "confidence": 0.90 + 0.05 * (i % 3),
                "label": "non_defective",
                "timestamp": f"t{i}",
            })

        # After filling the window, should not alert (same distribution)
        assert alert is None

    def test_alerts_on_low_confidence(self):
        """Should alert when confidence drops below threshold."""
        detector = DriftDetector(
            confidence_threshold=0.7,
            window_size=10,
            reference_confidence=[0.95] * 100
        )

        alert = None
        for i in range(15):
            alert = detector.log_prediction({
                "confidence": 0.55,  # Below threshold
                "label": "defective",
                "timestamp": f"t{i}",
            })

        assert alert is not None
        assert alert["severity"] in ["medium", "high"]

    def test_summary_reports_correctly(self):
        detector = DriftDetector(window_size=5)
        
        for i in range(10):
            detector.log_prediction({
                "confidence": 0.8,
                "label": "defective" if i % 2 == 0 else "non_defective",
                "timestamp": f"t{i}",
            })

        summary = detector.get_summary()
        assert summary["total_predictions"] == 10
        assert 0.0 <= summary["rolling_confidence_mean"] <= 1.0


class TestRetrainingTrigger:
    def test_no_retrain_when_metrics_good(self):
        trigger = RetrainingTrigger(
            confidence_threshold=0.70,
            baseline_accuracy=0.95,
        )

        decision = trigger.evaluate({
            "rolling_confidence_mean": 0.92,
            "monitored_accuracy": 0.94,
        })

        assert decision["should_retrain"] is False
        assert decision["urgency"] == "none"

    def test_retrain_on_accuracy_drop(self):
        trigger = RetrainingTrigger(
            confidence_threshold=0.70,
            accuracy_drop_threshold=0.05,
            baseline_accuracy=0.95,
        )

        decision = trigger.evaluate({
            "rolling_confidence_mean": 0.85,
            "monitored_accuracy": 0.85,  # 10% drop
        })

        assert decision["should_retrain"] is True
        assert any(t["trigger"] == "accuracy_drop" for t in decision["triggers_fired"])

    def test_retrain_on_sustained_low_confidence(self):
        trigger = RetrainingTrigger(
            confidence_threshold=0.70,
            confidence_consecutive_windows=3,
        )

        # Simulate 3 consecutive low-confidence windows
        for _ in range(3):
            decision = trigger.evaluate({
                "rolling_confidence_mean": 0.60,
            })

        assert decision["should_retrain"] is True
        assert any(t["trigger"] == "confidence_degradation" for t in decision["triggers_fired"])

    def test_streak_resets_on_recovery(self):
        trigger = RetrainingTrigger(
            confidence_threshold=0.70,
            confidence_consecutive_windows=3,
        )

        # 2 low windows
        trigger.evaluate({"rolling_confidence_mean": 0.60})
        trigger.evaluate({"rolling_confidence_mean": 0.65})
        
        # Recovery
        trigger.evaluate({"rolling_confidence_mean": 0.85})
        
        # Should not trigger yet
        decision = trigger.evaluate({"rolling_confidence_mean": 0.60})
        assert decision["should_retrain"] is False

    def test_urgency_levels(self):
        trigger = RetrainingTrigger(
            confidence_threshold=0.70,
            confidence_consecutive_windows=1,
            baseline_accuracy=0.95,
        )

        # Critical: both accuracy and confidence
        decision = trigger.evaluate({
            "rolling_confidence_mean": 0.50,
            "monitored_accuracy": 0.80,
        })
        assert decision["urgency"] == "critical"
