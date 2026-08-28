# Retraining Strategy Design Document

## Problem Statement

A deployed defect classification model will encounter distribution shift as manufacturing conditions change over time. Without a retraining strategy, model performance silently degrades, leading to missed defects (false negatives) or unnecessary production stops (false positives).

## Retraining Triggers

### Multi-Criteria Decision System

Retraining is triggered when ANY of the following conditions are met:

| Trigger | Condition | Priority | Rationale |
|---------|-----------|----------|-----------|
| Confidence Degradation | Rolling confidence < 0.70 for 3 consecutive windows | Medium | Model uncertainty indicates unfamiliar inputs |
| Accuracy Drop | Monitored accuracy drops > 5% from baseline | High | Direct performance regression |
| Distribution Shift | KS-test p-value < 0.01 | Medium | Statistical evidence of distributional change |
| Scheduled | Every 30 days | Low | Safety net for gradual drift |

### Why This Combination?

1. **Confidence alone is insufficient** — A model can be confidently wrong after certain types of drift
2. **Accuracy requires labels** — Not always immediately available; confidence fills the gap
3. **Statistical tests catch subtle shifts** — Useful for proactive retraining before degradation is visible
4. **Scheduled refresh is a safety net** — Catches drift that doesn't trigger other signals

## Retraining Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Trigger     │────▶│  Data        │────▶│  Retrain     │
│  Fired       │     │  Collection  │     │  Model       │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Deploy      │◀────│  A/B Test    │◀────│  Validate    │
│  (if pass)   │     │  (shadow)    │     │  New Model   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Step 1: Data Collection
- Collect recently predicted samples that triggered low confidence
- QA team labels a subset of recent predictions (active learning priority)
- Combine with original training data (ratio: 70% original, 30% new)

### Step 2: Model Retraining
- **Strategy: Fine-tune from current model** (not from scratch)
  - Justification: Preserves learned features, adapts to new patterns faster
  - Training on new data distribution while maintaining base knowledge
- Use same hyperparameters as best model from initial experiments
- Reduce learning rate to 1/10 of original (fine-tuning LR)
- Train for max 10 epochs with early stopping

### Step 3: Validation
- Evaluate on original held-out test set (must not regress)
- Evaluate on new samples (must show improvement)
- Metrics: Accuracy, Precision, Recall, F1 all must meet minimum thresholds

### Step 4: Shadow Deployment (A/B Testing)
- Deploy new model alongside current model
- Route 10% of traffic to new model (shadow mode)
- Compare metrics over 48 hours
- Promotion criteria: New model accuracy ≥ current model accuracy

### Step 5: Full Deployment
- If validation passes: swap models atomically
- Rollback plan: Previous model checkpoint always available
- Update model version tag in registry

## Urgency Levels

| Level | Response Time | Action |
|-------|--------------|--------|
| Critical | Same day | Accuracy + confidence both degraded; immediate retrain |
| High | 1-2 days | Accuracy dropped significantly; retrain with priority |
| Medium | 1 week | Confidence dropping; collect data and plan retrain |
| Low | Scheduled | Routine refresh; no urgency |

## Data Management for Retraining

- New samples stored in `data/retrain/` with timestamps
- Weighted sampling: Recent data gets 2x weight in training
- Maximum dataset growth: Cap at 2x original size (oldest non-original samples dropped)
- All data changes tracked via DVC

## Monitoring After Retrain

- Intensified monitoring for 72 hours post-deployment
- Alert thresholds temporarily tightened (0.80 instead of 0.70)
- Automatic rollback if metrics drop within first 24 hours

## Implementation

The retraining strategy is implemented in:
- `src/monitoring/drift_detector.py` — Drift detection signals
- `src/monitoring/retrain_strategy.py` — Trigger evaluation logic
- `src/monitoring/simulate_drift.py` — Validation via simulation

## Cost-Benefit Analysis

| Factor | Without Retraining | With Retraining |
|--------|-------------------|-----------------|
| Missed defects (monthly) | Increasing over time | Bounded |
| False alarms | Increasing over time | Bounded |
| Compute cost | Low | Moderate (retrain ~2h GPU) |
| Human labeling | None | ~100 samples/month |
| Quality assurance | Degrading | Maintained |

The cost of periodic retraining (~2 hours GPU + ~100 labels/month) is minimal compared to the cost of missed defects reaching customers.
