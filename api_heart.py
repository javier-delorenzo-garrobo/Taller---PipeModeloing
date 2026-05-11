import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from heart_model_wrapper import HeartDiseaseWrapper


APP_VERSION = "1.1.0"
DEFAULT_MODEL_PATH = "heart_disease_model.joblib"
DEFAULT_REGISTRY_PATH = "models/model_registry.json"


app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predecir enfermedad cardiaca usando Machine Learning",
    version=APP_VERSION,
)


def _csv_env(name: str, default: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:3000,http://127.0.0.1:3000",
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
wrapper = None
model_loaded_time = None
model_metadata: Dict[str, Any] = {}


PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Numero total de predicciones realizadas",
    ["endpoint"],
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Duracion de las predicciones en segundos",
    ["endpoint"],
)

ACTIVE_PREDICTIONS = Gauge(
    "active_predictions",
    "Numero de predicciones activas en este momento",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Numero total de errores en predicciones",
    ["endpoint"],
)

PREDICTION_RESULTS = Counter(
    "prediction_results_total",
    "Predicciones por etiqueta y nivel de riesgo",
    ["endpoint", "prediction_label", "risk_level"],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confianza devuelta por el modelo",
    ["endpoint", "prediction_label", "risk_level"],
    buckets=(0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0),
)

MODEL_LOADED = Gauge("model_loaded", "Indica si el modelo se cargo correctamente")
MODEL_FEATURES = Gauge("model_features", "Numero de variables usadas por el modelo")
MODEL_FILE_SIZE_BYTES = Gauge("model_file_size_bytes", "Tamano del archivo del modelo en bytes")
MODEL_LOAD_TIMESTAMP = Gauge("model_load_timestamp_seconds", "Timestamp Unix de carga del modelo")
MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Informacion de version del modelo activo",
    ["version", "sha256_short"],
)


def _initialize_metric_labels() -> None:
    for endpoint in ("predict", "predict_batch"):
        PREDICTIONS_TOTAL.labels(endpoint=endpoint).inc(0)
        PREDICTION_DURATION.labels(endpoint=endpoint)
        PREDICTION_ERRORS.labels(endpoint=endpoint).inc(0)
        for prediction_label in ("disease", "no_disease"):
            for risk_level in ("low", "medium", "high"):
                PREDICTION_RESULTS.labels(
                    endpoint=endpoint,
                    prediction_label=prediction_label,
                    risk_level=risk_level,
                ).inc(0)
                PREDICTION_CONFIDENCE.labels(
                    endpoint=endpoint,
                    prediction_label=prediction_label,
                    risk_level=risk_level,
                )


_initialize_metric_labels()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.exists():
        return {}
    with registry_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_model_metadata(model_path: Path) -> Dict[str, Any]:
    registry_path = Path(os.getenv("MODEL_REGISTRY_PATH", DEFAULT_REGISTRY_PATH))
    file_hash = _sha256(model_path)
    metadata: Dict[str, Any] = {
        "version": os.getenv("MODEL_VERSION", "unversioned"),
        "sha256": file_hash,
        "sha256_short": file_hash[:12],
        "path": str(model_path),
        "registry_path": str(registry_path),
        "registered": False,
    }

    registry = _load_registry(registry_path)
    versions = registry.get("versions", [])
    requested_version = os.getenv("MODEL_VERSION")
    active_version = registry.get("active_version")

    selected_entry = None
    if requested_version:
        selected_entry = next((entry for entry in versions if entry.get("version") == requested_version), None)
    if selected_entry is None and active_version:
        selected_entry = next((entry for entry in versions if entry.get("version") == active_version), None)
    if selected_entry is None:
        selected_entry = next((entry for entry in versions if entry.get("sha256") == file_hash), None)

    if selected_entry:
        metadata.update(selected_entry)
        metadata["registered"] = True
        metadata["sha256_short"] = selected_entry.get("sha256", file_hash)[:12]

    return metadata


def _attach_model_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    result["model_version"] = model_metadata.get("version", "unversioned")
    return result


def _metric_label(value: str) -> str:
    return value.lower().replace(" ", "_")


def _record_prediction_metrics(endpoint: str, result: Dict[str, Any]) -> None:
    prediction_label = _metric_label(result["prediction_label"])
    risk_level = _metric_label(result["risk_level"])
    PREDICTION_RESULTS.labels(
        endpoint=endpoint,
        prediction_label=prediction_label,
        risk_level=risk_level,
    ).inc()
    PREDICTION_CONFIDENCE.labels(
        endpoint=endpoint,
        prediction_label=prediction_label,
        risk_level=risk_level,
    ).observe(result["confidence"])


@app.on_event("startup")
async def load_model():
    global model, wrapper, model_loaded_time, model_metadata

    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    try:
        print(f"Cargando modelo desde {model_path}...")
        model = joblib.load(model_path)
        wrapper = HeartDiseaseWrapper(model)
        model_loaded_time = datetime.now()
        model_metadata = _resolve_model_metadata(model_path)

        MODEL_LOADED.set(1)
        MODEL_FEATURES.set(len(wrapper.feature_names))
        MODEL_FILE_SIZE_BYTES.set(model_path.stat().st_size)
        MODEL_LOAD_TIMESTAMP.set(time.time())
        MODEL_VERSION_INFO.labels(
            version=str(model_metadata.get("version", "unversioned")),
            sha256_short=str(model_metadata.get("sha256_short", "unknown")),
        ).set(1)

        print(
            "Modelo cargado correctamente "
            f"(version={model_metadata.get('version')}, sha256={model_metadata.get('sha256_short')})"
        )
    except Exception as exc:
        MODEL_LOADED.set(0)
        print(f"Error cargando modelo: {exc}")
        raise


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    chest: int = Field(..., ge=1, le=4)
    resting_blood_pressure: int = Field(..., ge=50, le=250)
    serum_cholestoral: int = Field(..., ge=100, le=600)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    resting_electrocardiographic_results: int = Field(..., ge=0, le=2)
    maximum_heart_rate_achieved: int = Field(..., ge=60, le=220)
    exercise_induced_angina: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    number_of_major_vessels: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    risk_level: str
    inference_time_ms: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData] = Field(..., min_items=1)


@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": APP_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "info": "/info",
            "model_version": "/model/version",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_time.isoformat() if model_loaded_time else None,
        "model_version": model_metadata.get("version", "unversioned"),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    steps = [step[0] for step in getattr(model, "steps", [])]

    return {
        "app_version": APP_VERSION,
        "model_type": str(type(model)),
        "model_steps": steps,
        "features": wrapper.feature_names,
        "n_features": len(wrapper.feature_names),
        "loaded_at": model_loaded_time.isoformat() if model_loaded_time else None,
        "model": model_metadata,
    }


@app.get("/model/version")
async def model_version():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_metadata


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_PREDICTIONS.inc()

    try:
        start_time = time.time()

        patient_dict = patient.dict()
        result = wrapper.predict(patient_dict)

        inference_time = time.time() - start_time
        result["inference_time_ms"] = round(inference_time * 1000, 2)
        result = _attach_model_metadata(result)

        PREDICTIONS_TOTAL.labels(endpoint="predict").inc()
        PREDICTION_DURATION.labels(endpoint="predict").observe(inference_time)
        _record_prediction_metrics("predict", result)

        return result

    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint="predict").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")
    finally:
        ACTIVE_PREDICTIONS.dec()


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_PREDICTIONS.inc()

    try:
        start_time = time.time()

        patients_list = [patient.dict() for patient in request.patients]
        results = [_attach_model_metadata(result) for result in wrapper.predict_batch(patients_list)]

        inference_time = time.time() - start_time

        PREDICTIONS_TOTAL.labels(endpoint="predict_batch").inc(len(results))
        PREDICTION_DURATION.labels(endpoint="predict_batch").observe(inference_time)
        for result in results:
            _record_prediction_metrics("predict_batch", result)

        return {
            "predictions": results,
            "count": len(results),
            "total_inference_time_ms": round(inference_time * 1000, 2),
            "avg_inference_time_ms": round(inference_time / len(results) * 1000, 2),
            "model_version": model_metadata.get("version", "unversioned"),
        }

    except Exception as exc:
        PREDICTION_ERRORS.labels(endpoint="predict_batch").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(exc)}")
    finally:
        ACTIVE_PREDICTIONS.dec()


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
