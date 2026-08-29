# Experiment Tracking Report (Deliverable 2)

**Tracking backend:** MLflow (SQLite: `mlflow.db`)
**Experiment name:** `defect_classification`
**Total tracked runs:** 1

## Model Comparison Table

| Rank | Architecture | Pretrained | Epochs | LR | Params | Test Acc | Precision | Recall | F1 | Val Acc | Run ID |
|------|-------------|-----------|--------|-----|--------|----------|-----------|--------|-----|---------|--------|
| 1 | resnet18 | False | 5 | 0.001 | 11177538 | 0.9963 | 0.9968 | 0.9968 | 0.9968 | 0.9972 | `55486444d7ab` |

## Best Model: resnet18

- **Test Accuracy:** 0.9963
- **F1 Score:** 0.9968
- **Run ID:** `55486444d7ab`

## Justification for Model Choice

The **ResNet18 (trained from scratch)** model was selected as the current production model:

1. **Offline reproducibility** — The run used `pretrained=False`, so it did not depend on downloading ImageNet weights and was reproducible in the current environment.
2. **Accuracy/speed balance** — ResNet18 (~11M params) offers near-best accuracy with faster inference than heavier models, suitable for production line throughput.
3. **Deployment maturity** — Well-supported for TorchScript export and containerized serving.

## Reproducibility

Any run can be reproduced from its logged configuration:
```bash
# View all runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Re-run training with the same config (seeds are fixed at 42)
python -m src.training.train --config configs/train_config.yaml
```

All hyperparameters, metrics, random seeds (42), and data splits are logged, enabling exact reconstruction of any experiment.