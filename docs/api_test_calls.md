# API Test Calls (Deliverable 3 + Demo Reference)

The API runs at `http://localhost:8000`. Start it with:

```bash
venv\Scripts\activate
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Interactive docs (Swagger UI): **http://localhost:8000/docs**

---

## 1. Health Check

**curl:**
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "uptime_seconds": 23.47
}
```

---

## 2. Model Info

**curl:**
```bash
curl http://localhost:8000/model-info
```

**Expected response:**
```json
{
  "architecture": "resnet18",
  "num_classes": 2,
  "input_size": 224,
  "model_path": "models/best_resnet18.pt",
  "device": "cpu"
}
```

---

## 3. Predict (Single Image)

**curl (Windows):**
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@data/splits/test/defective/cast_def_0_1000.png"
```

**Expected response:**
```json
{
  "prediction_id": "c3c6b1db-b81b-497d-8cf5-d59d91c8957b",
  "label": "defective",
  "confidence": 1.0,
  "defective": true,
  "inference_time_ms": 25.8,
  "timestamp": "2026-08-06T15:52:32"
}
```

---

## 4. Edge Case — Invalid File Type (Error Handling)

**curl:**
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@README.md"
```

**Expected response (400):**
```json
{
  "detail": "Invalid file type: text/markdown. Accepted: image/jpeg, image/png, image/bmp"
}
```

---

## 5. Recent Predictions (Monitoring)

**curl:**
```bash
curl http://localhost:8000/predictions/recent?limit=10
```

---

## Postman Collection

Import `docs/postman_collection.json` into Postman to test all endpoints with a GUI.

## Python test snippet

```python
import requests

# Health
print(requests.get("http://localhost:8000/health").json())

# Predict
with open("data/splits/test/defective/cast_def_0_1000.png", "rb") as f:
    r = requests.post("http://localhost:8000/predict",
                      files={"file": ("test.png", f, "image/png")})
    print(r.json())
```
