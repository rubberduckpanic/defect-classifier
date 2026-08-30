"""
Module M4: FastAPI Model Serving
Production-style REST API for defect classification inference.

Features:
- POST /predict: Upload image, get defect classification
- GET /health: Health check endpoint
- GET /model-info: Model metadata
- Input validation and error handling
- Prediction logging for monitoring (M5)
"""

import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from src.monitoring.drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Application
app = FastAPI(
    title="Defect Classifier API",
    description="Image-based defect detection for manufacturing quality assurance",
    version="1.0.0",
)

# CORS middleware for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    prediction_id: str
    label: str
    confidence: float
    defective: bool
    inference_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes: int
    input_size: int
    model_path: str
    device: str


# Global model state
class ModelService:
    """Encapsulates model loading and inference logic."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.transform: Optional[transforms.Compose] = None
        self.model_path: str = ""
        self.architecture: str = ""
        self.class_names = ["ok_front", "def_front"]
        self.start_time = time.time()
        self.prediction_log: list = []
        self.prediction_log_path = Path(
            os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.jsonl")
        )
        self.drift_detector = DriftDetector()

    def _persist_prediction(self, prediction: dict) -> None:
        """Persist a prediction record and feed it to drift monitoring."""
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.prediction_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(prediction) + "\n")
        self.drift_detector.log_prediction(prediction)

    def load_model(self, model_path: str = "models/best_resnet18.pt"):
        """Load trained model from checkpoint."""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(model_path).exists():
            # Try TorchScript model
            ts_path = model_path.replace(".pt", "_scripted.pt")
            if Path(ts_path).exists():
                self.model = torch.jit.load(ts_path, map_location=self.device)
                self.architecture = "torchscript"
                logger.info(f"Loaded TorchScript model from {ts_path}")
            else:
                logger.warning(f"Model not found at {model_path}. API will return errors.")
                return
        else:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Reconstruct model from config
            config = checkpoint.get("config", {})
            model_config = config.get("model", {})
            self.architecture = model_config.get("architecture", "resnet18")

            from src.training.models import get_model
            self.model = get_model(
                architecture=self.architecture,
                num_classes=model_config.get("num_classes", 2),
                pretrained=False,
                dropout=model_config.get("dropout", 0.3)
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            logger.info(f"Loaded {self.architecture} from {model_path}")

        self.model.eval()

        # Setup inference transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Prediction dictionary with label, confidence, timing
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Preprocess
        start_time = time.time()
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        inference_time = (time.time() - start_time) * 1000  # ms

        label = self.class_names[predicted_idx.item()]
        conf = confidence.item()

        # Log prediction for monitoring
        prediction_record = {
            "prediction_id": str(uuid.uuid4()),
            "label": label,
            "confidence": conf,
            "defective": label == "def_front",
            "inference_time_ms": inference_time,
            "timestamp": datetime.utcnow().isoformat(),
            "probabilities": probabilities[0].cpu().numpy().tolist(),
            "image_brightness": float(np.asarray(image).mean() / 255.0),
            "image_contrast": float(np.asarray(image).std() / 255.0),
        }
        self.prediction_log.append(prediction_record)
        self._persist_prediction(prediction_record)

        return prediction_record


# Initialize service
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    model_path = os.environ.get("MODEL_PATH", "models/best_resnet18.pt")
    try:
        model_service.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy" if model_service.model is not None else "degraded",
        model_loaded=model_service.model is not None,
        device=str(model_service.device),
        uptime_seconds=time.time() - model_service.start_time,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    return ModelInfoResponse(
        architecture=model_service.architecture,
        num_classes=2,
        input_size=224,
        model_path=model_service.model_path,
        device=str(model_service.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image as defective or non-defective.
    
    Accepts: JPEG, PNG, BMP image files.
    Returns: Classification label, confidence score, and inference timing.
    
    Example:
        curl -X POST "http://localhost:8000/predict" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@casting_image.png"
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Accepted: image/jpeg, image/png, image/bmp"
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size: 10MB"
        )

    # Validate image can be opened
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image. File may be corrupt."
        )

    # Validate minimum dimensions
    if image.size[0] < 32 or image.size[1] < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({image.size[0]}x{image.size[1]}). "
                   f"Minimum: 32x32 pixels"
        )

    # Check model is loaded
    if model_service.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up."
        )

    # Run inference
    try:
        result = model_service.predict(image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal inference error"
        )

    return PredictionResponse(
        prediction_id=result["prediction_id"],
        label=result["label"],
        confidence=result["confidence"],
        defective=result["defective"],
        inference_time_ms=result["inference_time_ms"],
        timestamp=result["timestamp"],
    )


@app.get("/predictions/recent")
async def recent_predictions(limit: int = 50):
    """Get recent prediction logs for monitoring."""
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    return model_service.prediction_log[-limit:]


@app.get("/monitoring/summary")
async def monitoring_summary():
    """Return the current drift monitoring summary."""
    return model_service.drift_detector.get_summary()


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction endpoint for multiple images.
    Limited to 16 images per request.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(files) > 16:
        raise HTTPException(
            status_code=400,
            detail="Maximum 16 images per batch request"
        )

    results = []
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
            results.append({"error": "Invalid file type", "filename": file.filename})
            continue
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            results.append({"error": "File too large", "filename": file.filename})
            continue
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            if image.size[0] < 32 or image.size[1] < 32:
                results.append({"error": "Image too small", "filename": file.filename})
                continue
            if model_service.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            result = model_service.predict(image)
            results.append(result)
        except HTTPException:
            raise
        except Exception:
            results.append({"error": "Cannot decode or infer image", "filename": file.filename})

    return {"predictions": results, "total": len(results)}
