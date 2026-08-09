# Image-Based Defect / Quality Classifier

An end-to-end ML system that automatically flags defective products from images captured on a manufacturing production line. The pipeline ingests and preprocesses product images, trains a classifier to distinguish defective from non-defective items, deploys it as an inference service, and monitors performance as new product variants or lighting conditions appear.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ML System Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────┐    ┌────────────────┐    ┌────────────────┐    ┌─────────────┐  │
│  │   Data      │    │  Experiment    │    │    Model       │    │ Monitoring  │  │
│  │  Pipeline   │───▶│  Tracking      │───▶│   Serving      │───▶│  & Drift    │  │
│  │   (M2)      │    │   (M3)         │    │    (M4)        │    │   (M5)      │  │
│  └────────────┘    └────────────────┘    └────────────────┘    └─────────────┘  │
│       │                   │                      │                     │         │
│       ▼                   ▼                      ▼                     ▼         │
│  ┌────────────┐    ┌────────────────┐    ┌────────────────┐    ┌─────────────┐  │
│  │ DVC         │    │ MLflow         │    │ FastAPI        │    │ Drift       │  │
│  │ Versioned   │    │ Experiments    │    │ + Docker       │    │ Detection   │  │
│  │ Dataset     │    │ & Registry     │    │ Container      │    │ & Retrain   │  │
│  └────────────┘    └────────────────┘    └────────────────┘    └─────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
defect-classifier/
├── data/                    # Data directory (DVC tracked)
│   ├── raw/                 # Original dataset images
│   ├── processed/           # Preprocessed (resized, RGB) images
│   └── splits/              # Train/val/test stratified splits
├── src/
│   ├── data/                # M2: Ingestion, validation, preprocessing
│   │   ├── ingest.py        # Download from Kaggle or local source
│   │   ├── validate.py      # Image integrity & quality checks
│   │   └── preprocess.py    # Resize, normalize, split
│   ├── training/            # M3: Model training & experiments
│   │   ├── models.py        # CNN Baseline, ResNet18, EfficientNet-B0
│   │   ├── dataset.py       # PyTorch Dataset & DataLoaders
│   │   ├── train.py         # Training loop with MLflow tracking
│   │   └── run_experiments.py  # Multi-model comparison runner
│   ├── serving/             # M4: Production API
│   │   ├── app.py           # FastAPI REST endpoints
│   │   └── export_model.py  # TorchScript/ONNX export & benchmarking
│   └── monitoring/          # M5: Drift & retraining
│       ├── drift_detector.py    # Multi-signal drift detection
│       ├── simulate_drift.py    # Drift simulation experiments
│       └── retrain_strategy.py  # Retraining trigger logic
├── configs/
│   └── train_config.yaml    # All hyperparameters (single source of truth)
├── models/                  # Saved model checkpoints
├── docs/                    # Reports & documentation
│   ├── model_comparison.md  # Experiment comparison report
│   ├── drift_report.md      # Drift simulation findings
│   └── retraining_design.md # Retraining strategy document
├── logs/                    # Monitoring & prediction logs
├── notebooks/               # Exploration notebooks
├── dvc.yaml                 # DVC pipeline DAG
├── Dockerfile               # Production container
├── docker-compose.yml       # Multi-service deployment
├── requirements.txt         # Python dependencies (pinned versions)
├── Makefile                 # Common commands
└── README.md
```

## Dataset

**Casting Product Image Data for Quality Inspection** (Kaggle)
- Source: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- Binary classification: defective vs non-defective casting products
- ~7000 images (300x300 grayscale)
- License: CC0 Public Domain

## Setup Instructions to follow

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git
- DVC (`pip install dvc`)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd defect-classifier

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize DVC (if not already)
dvc init

# 5. Configure Kaggle API (for data download)
# Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME & KAGGLE_KEY
```

### Running the Pipeline

```bash
# Option A: Full pipeline via DVC
dvc repro

# Option B: Step by step
python -m src.data.ingest --output data/raw
python -m src.data.validate --data-dir data/raw
python -m src.data.preprocess --config configs/train_config.yaml
python -m src.training.train --config configs/train_config.yaml
python -m src.serving.export_model --checkpoint models/best_resnet18.pt
```

### Running Experiments

```bash
# Train all model variants and generate comparison report
python -m src.training.run_experiments

# View results in MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

### Starting the API

```bash
# Option A: Direct
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Option B: Docker
docker-compose up --build
```

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.png"

# Response format:
# {
#   "prediction_id": "uuid",
#   "label": "defective",
#   "confidence": 0.94,
#   "defective": true,
#   "inference_time_ms": 15.2,
#   "timestamp": "2024-01-15T10:30:00"
# }
```

### Running Drift Simulation

```bash
# Simulate lighting drift
python -m src.monitoring.simulate_drift --drift-type lighting --steps 50

# Simulate mixed drift
python -m src.monitoring.simulate_drift --drift-type mixed --steps 50

# View retraining trigger demo
python -m src.monitoring.retrain_strategy
```

## Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| ML Framework | PyTorch | Industry standard for vision, strong transfer-learning support |
| Experiment Tracking | MLflow | Open-source, model registry, easy metric comparison |
| Data Versioning | DVC | Git-like workflow for large binary files, pipeline DAG |
| API Framework | FastAPI | Async support, auto-docs (Swagger), Pydantic validation |
| Containerization | Docker | Reproducible deployment, environment isolation |
| Model Architecture | ResNet18 | Best accuracy/speed tradeoff for binary classification |
| Drift Detection | Multi-signal (confidence + KS-test + input features) | Robust detection; no single signal catches all drift types |
| Retraining | Fine-tune from current model | Faster adaptation, preserves learned features |
| Export Format | TorchScript | No Python dependency at inference, JIT optimization |

## Module Details

### M2 — Data Engineering & Versioning (Week 1)
- Automated Kaggle download with fallback to local ingestion
- Image validation: format, corruption, size, duplicate detection
- Preprocessing: RGB conversion, resize to 224x224, LANCZOS resampling
- Stratified train/val/test split (70/15/15) maintaining class balance
- Dataset versioning via DVC with pipeline DAG

### M3 — Experimentation & Reproducibility (Week 2)
- 4 experiments: CNN baseline, ResNet18 (2 variants), EfficientNet-B0
- Full MLflow tracking: hyperparams, epoch metrics, artifacts, models
- Early stopping with patience=5 on validation accuracy
- Reproducible via logged configs and fixed random seeds
- Comparison report with quantitative analysis

### M4 — Model Packaging & Deployment (Week 3)
- TorchScript export with output verification
- FastAPI with input validation (file type, size, dimensions)
- Batch prediction endpoint (up to 16 images)
- Docker multi-stage build (slim production image)
- Inference benchmarking (latency percentiles, throughput)

### M5 — Monitoring, Drift & Retraining (Week 4)
- Prediction logging (confidence, labels, timestamps)
- Multi-signal drift detection (confidence, KS-test, input features)
- 4 drift simulation scenarios (lighting, angle, noise, mixed)
- Documented retraining trigger with urgency levels
- A/B testing workflow design for safe model updates

## Third-Party Libraries

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| PyTorch | 2.1.0 | Deep learning framework | BSD |
| torchvision | 0.16.0 | Vision models & transforms | BSD |
| MLflow | 2.9.1 | Experiment tracking | Apache 2.0 |
| DVC | 3.30.1 | Data versioning | Apache 2.0 |
| FastAPI | 0.104.1 | REST API framework | MIT |
| scikit-learn | 1.3.2 | Metrics & splitting | BSD |
| Pillow | 10.1.0 | Image processing | HPND |
| scipy | 1.11.4 | Statistical tests | BSD |
| evidently | 0.4.10 | ML monitoring | Apache 2.0 |

## License

Academic project. All third-party libraries and datasets are used under their respective open-source licenses as cited above.
