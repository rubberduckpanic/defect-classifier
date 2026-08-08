.PHONY: setup data train serve monitor test clean docker-build docker-up

# Setup environment
setup:
	python -m venv venv
	pip install -r requirements.txt
	dvc init
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &

# Download and prepare data
data:
	python -m src.data.ingest
	python -m src.data.validate
	python -m src.data.preprocess
	dvc add data/raw
	dvc add data/processed

# Train model
train:
	python -m src.training.train --config configs/train_config.yaml

# Run experiments (multiple models)
experiments:
	python -m src.training.run_experiments

# Serve model via API
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Run monitoring
monitor:
	python -m src.monitoring.drift_detector
	python -m src.monitoring.simulate_drift

# Run tests
test:
	pytest tests/ -v

# Build Docker image
docker-build:
	docker build -t defect-classifier:latest .

# Start all services
docker-up:
	docker-compose up --build -d

# Clean artifacts
clean:
	rm -rf mlruns/ models/*.pt data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +
