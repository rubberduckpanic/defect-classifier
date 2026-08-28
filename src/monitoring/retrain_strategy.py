"""
Module M5: Retraining Strategy
Defines automated retraining trigger logic and workflow.

Retraining Trigger Design:
─────────────────────────────────
The retraining workflow is triggered based on a combination of signals:

1. CONFIDENCE DEGRADATION TRIGGER
   - Condition: Rolling mean confidence < 0.70 for 3 consecutive windows
   - Rationale: Sustained low confidence indicates the model is uncertain about
     inputs it encounters, suggesting the data distribution has shifted.

2. ACCURACY DRIFT TRIGGER  
   - Condition: Monitored accuracy drops > 5% from baseline
   - Rationale: Direct performance degradation requires immediate action.
   - Note: Requires ground-truth labels (available via QA sampling in manufacturing)

3. DISTRIBUTION SHIFT TRIGGER
   - Condition: KS-test p-value < 0.01 for confidence distributions
   - Rationale: Statistical evidence of distributional change, even if accuracy
     hasn't degraded yet (proactive retraining).

4. SCHEDULED TRIGGER (Safety net)
   - Condition: Every 30 days regardless of drift signals
   - Rationale: Ensures model freshness; catches gradual drift that individual
     signals might miss.

Retraining Workflow:
1. Collect new labeled samples (triggered by alert or schedule)
2. Combine with original training data (weighted towards recent)
3. Fine-tune from current best model (not from scratch)
4. Validate on held-out test set + recent samples
5. A/B test: compare new vs current model on shadow traffic
6. Deploy if new model metrics >= current model metrics
7. Log all decisions for auditability
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Evaluates whether retraining should be triggered based on monitoring signals.
    
    Implements a multi-criteria decision system with configurable thresholds.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        confidence_consecutive_windows: int = 3,
        accuracy_drop_threshold: float = 0.05,
        ks_pvalue_threshold: float = 0.01,
        scheduled_interval_days: int = 30,
        baseline_accuracy: float = 0.95,
    ):
        self.confidence_threshold = confidence_threshold
        self.confidence_consecutive_windows = confidence_consecutive_windows
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.scheduled_interval_days = scheduled_interval_days
        self.baseline_accuracy = baseline_accuracy

        # State tracking
        self.low_confidence_streak = 0
        self.last_retrain_date: Optional[datetime] = None
        self.retrain_history: List[Dict] = []
        self.current_model_version = "v1.0"

    def evaluate(self, monitoring_summary: Dict) -> Dict:
        """
        Evaluate whether retraining should be triggered.
        
        Args:
            monitoring_summary: Current monitoring state from DriftDetector
            
        Returns:
            Decision dict with trigger status and reasoning
        """
        triggers_fired = []
        
        # Check 1: Confidence degradation
        rolling_confidence = monitoring_summary.get("rolling_confidence_mean", 1.0)
        if rolling_confidence < self.confidence_threshold:
            self.low_confidence_streak += 1
        else:
            self.low_confidence_streak = 0

        if self.low_confidence_streak >= self.confidence_consecutive_windows:
            triggers_fired.append({
                "trigger": "confidence_degradation",
                "value": rolling_confidence,
                "threshold": self.confidence_threshold,
                "consecutive_windows": self.low_confidence_streak,
            })

        # Check 2: Accuracy drop (if accuracy is monitored)
        current_accuracy = monitoring_summary.get("monitored_accuracy")
        if current_accuracy is not None:
            drop = self.baseline_accuracy - current_accuracy
            if drop > self.accuracy_drop_threshold:
                triggers_fired.append({
                    "trigger": "accuracy_drop",
                    "current_accuracy": current_accuracy,
                    "baseline_accuracy": self.baseline_accuracy,
                    "drop": drop,
                })

        # Check 3: Distribution shift
        last_alert = monitoring_summary.get("last_alert")
        if last_alert:
            for signal in last_alert.get("signals", []):
                if signal.get("signal") == "distribution_shift":
                    if signal.get("p_value", 1.0) < self.ks_pvalue_threshold:
                        triggers_fired.append({
                            "trigger": "distribution_shift",
                            "p_value": signal["p_value"],
                            "ks_statistic": signal["ks_statistic"],
                        })

        # Check 4: Scheduled retrain
        now = datetime.utcnow()
        if self.last_retrain_date:
            days_since = (now - self.last_retrain_date).days
            if days_since >= self.scheduled_interval_days:
                triggers_fired.append({
                    "trigger": "scheduled",
                    "days_since_last_retrain": days_since,
                    "interval": self.scheduled_interval_days,
                })
        else:
            # No retrain history — assume initial deployment was recent
            pass

        # Decision
        should_retrain = len(triggers_fired) > 0
        decision = {
            "timestamp": now.isoformat(),
            "should_retrain": should_retrain,
            "triggers_fired": triggers_fired,
            "num_triggers": len(triggers_fired),
            "urgency": self._assess_urgency(triggers_fired),
            "recommended_action": self._get_action(triggers_fired),
            "model_version": self.current_model_version,
        }

        if should_retrain:
            logger.warning(f"RETRAIN DECISION: YES — {len(triggers_fired)} triggers fired")
            for t in triggers_fired:
                logger.warning(f"  - {t['trigger']}")
        else:
            logger.info("RETRAIN DECISION: No — all metrics within bounds")

        return decision

    def _assess_urgency(self, triggers: List[Dict]) -> str:
        """Assess urgency level of retraining need."""
        if not triggers:
            return "none"
        
        trigger_types = [t["trigger"] for t in triggers]
        
        if "accuracy_drop" in trigger_types and "confidence_degradation" in trigger_types:
            return "critical"
        elif "accuracy_drop" in trigger_types:
            return "high"
        elif "confidence_degradation" in trigger_types:
            return "medium"
        elif "scheduled" in trigger_types:
            return "low"
        else:
            return "medium"

    def _get_action(self, triggers: List[Dict]) -> str:
        """Get recommended action based on triggers."""
        if not triggers:
            return "Continue monitoring. No action required."

        trigger_types = [t["trigger"] for t in triggers]

        if "accuracy_drop" in trigger_types:
            return (
                "IMMEDIATE: Accuracy has dropped significantly. "
                "1) Collect recent mislabeled samples. "
                "2) Fine-tune from current checkpoint with new data. "
                "3) Validate on expanded test set. "
                "4) Deploy with A/B testing."
            )
        elif "confidence_degradation" in trigger_types:
            return (
                "SOON: Model confidence degrading. "
                "1) Sample recent low-confidence predictions for review. "
                "2) Label new edge cases. "
                "3) Retrain with augmented dataset. "
                "4) Shadow-test before full deployment."
            )
        elif "distribution_shift" in trigger_types:
            return (
                "PROACTIVE: Distribution shift detected. "
                "1) Investigate root cause (lighting? new product variant?). "
                "2) Collect representative samples of new distribution. "
                "3) Retrain with balanced old + new data."
            )
        else:
            return (
                "SCHEDULED: Routine refresh. "
                "1) Accumulate any new labeled data. "
                "2) Retrain with updated dataset. "
                "3) Standard validation pipeline."
            )

    def record_retrain(self, metrics: Dict) -> None:
        """Record a completed retraining event."""
        self.last_retrain_date = datetime.utcnow()
        self.low_confidence_streak = 0
        self.current_model_version = metrics.get("new_version", "v_unknown")

        self.retrain_history.append({
            "timestamp": self.last_retrain_date.isoformat(),
            "new_version": self.current_model_version,
            "metrics": metrics,
        })

    def save_state(self, output_path: str = "logs/retrain_state.json"):
        """Save trigger state and history."""
        state = {
            "current_model_version": self.current_model_version,
            "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            "low_confidence_streak": self.low_confidence_streak,
            "retrain_history": self.retrain_history,
            "config": {
                "confidence_threshold": self.confidence_threshold,
                "consecutive_windows": self.confidence_consecutive_windows,
                "accuracy_drop_threshold": self.accuracy_drop_threshold,
                "ks_pvalue_threshold": self.ks_pvalue_threshold,
                "scheduled_interval_days": self.scheduled_interval_days,
            }
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Retrain state saved to {output_path}")


if __name__ == "__main__":
    # Demo: Simulate monitoring data and trigger evaluation
    trigger = RetrainingTrigger(
        confidence_threshold=0.70,
        baseline_accuracy=0.95,
    )

    # Scenario: Confidence has been dropping
    monitoring_states = [
        {"rolling_confidence_mean": 0.85, "monitored_accuracy": 0.94},
        {"rolling_confidence_mean": 0.75, "monitored_accuracy": 0.92},
        {"rolling_confidence_mean": 0.68, "monitored_accuracy": 0.88},
        {"rolling_confidence_mean": 0.65, "monitored_accuracy": 0.85},
        {"rolling_confidence_mean": 0.62, "monitored_accuracy": 0.82},
    ]

    print("\n" + "=" * 60)
    print("RETRAINING TRIGGER SIMULATION")
    print("=" * 60)

    for i, state in enumerate(monitoring_states):
        print(f"\n--- Window {i+1} ---")
        print(f"  Confidence: {state['rolling_confidence_mean']:.2f}")
        print(f"  Accuracy: {state.get('monitored_accuracy', 'N/A')}")
        
        decision = trigger.evaluate(state)
        print(f"  Should retrain: {decision['should_retrain']}")
        print(f"  Urgency: {decision['urgency']}")
        if decision['triggers_fired']:
            for t in decision['triggers_fired']:
                print(f"    Trigger: {t['trigger']}")
