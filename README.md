# Defect Detection Classifier — ML Engineering Mini-Project

An end-to-end machine learning pipeline for automatically detecting defective products from production-line images.

## Project Overview

This project implements an **Image-Based Defect / Quality Classifier** as part of the ML Engineering (PCAM* ZC412) course. It covers the full ML system lifecycle:

1. **Data Engineering & Versioning** — Image ingestion, validation, preprocessing/augmentation, and dataset versioning with DVC
2. **Experimentation & Reproducibility** — CNN and transfer-learning model training with MLflow experiment tracking
3. **Model Packaging & Deployment** — REST API service via FastAPI, containerized with Docker
4. **Monitoring & Drift Detection** — Prediction logging, distribution shift simulation, and retraining trigger design

## Project Structure

```
defect-classifier/
├── configs/
│   ├── data_config.yaml     # Dataset paths, split ratios, augmentation settings
│   ├── train_config.yaml    # Model choice, hyperparameters, MLflow config
│   └── serve_config.yaml    # API port, model path, monitoring thresholds
├── src/
│   ├── data/                # Layer 1: Data ingestion, validation, preprocessing
│   │   ├── ingest.py        #   Download and organize dataset
│   │   ├── validate.py      #   Check for corrupt/invalid images
│   │   ├── preprocess.py    #   Resize, normalize, standardize format
│   │   └── augment.py       #   Training-time augmentations (Albumentations)
│   ├── training/            # Layer 2: Model training and evaluation
│   │   ├── dataset.py       #   PyTorch Dataset class
│   │   ├── models.py        #   CNN, ResNet-18, EfficientNet-B0 architectures
│   │   ├── train.py         #   Training loop with MLflow tracking
│   │   └── evaluate.py      #   Metrics, confusion matrix, model comparison
│   ├── serving/             # Layer 3: REST API for inference
│   │   ├── app.py           #   FastAPI application (endpoints)
│   │   ├── inference.py     #   Model loading and prediction logic
│   │   └── schemas.py       #   Pydantic request/response models
│   └── monitoring/          # Layer 4: Production monitoring
│       ├── logger.py        #   Structured prediction logging
│       ├── drift.py         #   Distribution shift detection
│       ├── simulate_drift.py#   Artificial drift for testing
│       └── retrain.py       #   Retraining trigger logic
├── tests/                   # Unit and integration tests
├── notebooks/               # EDA and model comparison notebooks
├── models/                  # Saved model artifacts (.pt files)
├── docs/                    # Architecture, model report, drift report
├── data/
│   ├── raw/                 # Original images (DVC tracked, gitignored)
│   └── processed/           # Preprocessed train/val/test splits
├── Dockerfile               # Container image for the API service
├── docker-compose.yml       # Multi-service setup (API + MLflow)
├── requirements.txt         # Pinned Python dependencies
└── .gitignore
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker |
| Drift Detection | Evidently AI / scipy |
| Image Processing | Pillow, OpenCV, Albumentations |
| Testing | pytest |

## Dataset

**Casting Product Image Data for Quality Inspection** from Kaggle.

- ~7000 images (512x512 grayscale)
- Binary classification: `def_front` (defective) vs `ok_front` (non-defective)
- Source: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

## Setup

```bash
# Clone and set up environment
git clone https://github.com/<your-org>/defect-classifier.git
cd defect-classifier
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Pull data
dvc pull

# Run the pipeline
python -m src.data.ingest
python -m src.data.validate
python -m src.data.preprocess
python -m src.training.train --config configs/train_config.yaml

# Start the API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Or run via Docker
docker-compose up --build
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict -F "file=@test_image.png"
```

Response:
```json
{
  "prediction": "defective",
  "confidence": 0.94,
  "model_version": "v1.0",
  "timestamp": "2026-08-10T14:30:00Z"
}
```

## License

This project is for academic purposes as part of BITS Pilani ML Engineering coursework.
