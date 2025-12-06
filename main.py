"""
FastAPI ML microservice scaffold for Sleep Environment Scoring
- Endpoints implemented (stubs + rule-based scoring):
  POST /v1/score/interval
  POST /v1/score/interval/batch
  POST /v1/score/session
  GET  /v1/health
  GET  /v1/model/info
  POST /v1/model/reload
  POST /v1/debug/score  (dev only)

How to use:
1. Install dependencies:
   pip install fastapi uvicorn pydantic joblib scikit-learn numpy
2. Run:
   uvicorn fastapi_ml_scaffold:app --host 0.0.0.0 --port 8000 --reload
3. Drop your trained residual model artifact (joblib) at the path configured in MODEL_CONFIG.

Notes:
- This scaffold is intentionally minimal and synchronous-friendly. The model is loaded into memory at startup.
- The rule-based scorer is deterministic and used as fallback when model unavailable.
- Extend compute_features() if you want richer engineered features.

"""
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
import joblib
import os
import logging

# -----------------------------
# Configuration
# -----------------------------
MODEL_CONFIG = {
    "artifact_path": os.environ.get("MODEL_ARTIFACT", "models/residual_model.joblib"),
    "version": None
}

API_KEY_HEADER = "X-API-Key"
ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "admin-secret")
# Expected ranges for model.info
EXPECTED_RANGES = {
    "temperature": [15.0, 40.0],
    "humidity": [20.0, 100.0],
    "light": [0.0, 2000.0],
    "sound_level": [15.0, 120.0]
}

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sleep-ml")

# -----------------------------
# Simple rule-based scorer (deterministic)
# -----------------------------
TEMP_IDEAL = 20.0
HUM_IDEAL = 50.0
NOISE_THRESHOLD = 30.0

rng = np.random.default_rng(42)


def light_penalty_from_lux(lux: float, sensor_lower_limit: Optional[float] = None) -> float:
    if lux is None:
        return 0.0
    if sensor_lower_limit is not None and lux == 0:
        lux = sensor_lower_limit
    if lux <= 1.0:
        return 0.0
    if lux <= 10.0:
        return 1.0 * (lux - 1.0)
    base = 1.0 * (10.0 - 1.0)
    extra = 0.6 * (lux - 10.0)
    return base + extra


def rule_score(temp: float, hum: float, noise_db: float, light: float) -> float:
    temp_pen = 3.5 * abs(temp - TEMP_IDEAL)
    hum_pen = 0.4 * abs(hum - HUM_IDEAL)
    noise_pen = 2.0 * max(0.0, noise_db - NOISE_THRESHOLD)
    light_pen = 1.5 * light_penalty_from_lux(light, sensor_lower_limit=0.5)
    score = 100.0 - (temp_pen + hum_pen + noise_pen + light_pen)
    return float(np.clip(score, 0.0, 100.0))


# -----------------------------
# Model wrapper
# -----------------------------
class ModelWrapper:
    def __init__(self, artifact_path: str):
        self.artifact_path = artifact_path
        self.model = None
        self.version = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.artifact_path):
            try:
                self.model = joblib.load(self.artifact_path)
                # infer a version string from filename or model attr if available
                self.version = getattr(self.model, "version", os.path.basename(self.artifact_path))
                logger.info(f"Loaded model from {self.artifact_path} version={self.version}")
            except Exception as e:
                logger.exception("Failed to load model artifact")
                self.model = None
                self.version = None
        else:
            logger.warning(f"Model artifact not found at {self.artifact_path}")
            self.model = None
            self.version = None

    def predict_residual(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded")
        # expect model to take 2D array
        return self.model.predict(X)


model_service = ModelWrapper(MODEL_CONFIG["artifact_path"])

# -----------------------------
# Pydantic schemas
# -----------------------------
class IntervalRecord(BaseModel):
    session_id: Optional[str] = Field(None, description="optional session id")
    device_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    light: float
    sound_level: float
    sample_count: Optional[int] = None


class IntervalScoreResponse(BaseModel):
    interval_score: float
    rule_score: float
    residual: Optional[float]
    model_version: Optional[str]
    confidence: Optional[float]
    timestamp: datetime


class BatchRequest(BaseModel):
    records: List[IntervalRecord]


class BatchResponseItem(BaseModel):
    timestamp: datetime
    interval_score: float
    model_version: Optional[str]
    confidence: Optional[float]


class BatchResponse(BaseModel):
    results: List[BatchResponseItem]
    processed: int


class SessionRequest(BaseModel):
    session_id: str
    device_id: str
    start_ts: datetime
    end_ts: datetime
    total_sleep_duration_minutes: int
    avg_temperature: float
    avg_humidity: float
    avg_light: float
    avg_sound_level: float
    interval_scores: Optional[List[float]] = None
    num_intervals: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_score: float
    rule_session_score: float
    residual_session: Optional[float]
    model_version: Optional[str]
    breakdown: Dict[str, float]


class ModelInfo(BaseModel):
    model_version: Optional[str]
    trained_at: Optional[str] = None
    artifact_uri: Optional[str] = None
    features: List[str]
    expected_ranges: Dict[str, List[float]]
    notes: Optional[str] = None


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Sleep ML Scoring Service", version="0.1.0")


def check_api_key(x_api_key: Optional[str]):
    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    # In production, validate against DB or secret store
    return True


# Feature assembler (simple here, replace with your feature engineering)
def compute_features_from_interval(r: IntervalRecord) -> np.ndarray:
    # simple features: [temp, hum, light, sound]
    return np.array([[r.temperature, r.humidity, r.light, r.sound_level]])


# Confidence estimation (very simple heuristic)
def estimate_confidence(rule_val: float, residual_pred: Optional[float]) -> float:
    # higher confidence when residual magnitude small
    if residual_pred is None:
        return 0.5
    score = 1.0 - min(1.0, abs(residual_pred) / 10.0)
    return float(np.clip(score, 0.0, 1.0))


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/v1/score/interval", response_model=IntervalScoreResponse)
async def score_interval(record: IntervalRecord, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    # basic validation and clipping
    for k, (lo, hi) in EXPECTED_RANGES.items():
        val = getattr(record, k if k != 'sound_level' else 'sound_level')
        # skip detailed enforcement; just warn
        if val is None:
            logger.debug(f"Missing {k} in request")

    # compute rule-based score
    r_score = rule_score(record.temperature, record.humidity, record.sound_level, record.light)

    # model residual prediction if available
    residual = None
    model_ver = model_service.version
    confidence = None
    try:
        feats = compute_features_from_interval(record)
        if model_service.model is not None:
            pred = model_service.predict_residual(feats)
            residual = float(pred[0])
            confidence = estimate_confidence(r_score, residual)
        else:
            model_ver = None
            residual = None
            confidence = 0.5
    except Exception as e:
        logger.exception("Model prediction failed")
        residual = None
        model_ver = None
        confidence = 0.0

    interval_score = float(np.clip(r_score + (residual if residual is not None else 0.0), 0.0, 100.0))

    return IntervalScoreResponse(
        interval_score=interval_score,
        rule_score=float(r_score),
        residual=residual,
        model_version=model_ver,
        confidence=confidence,
        timestamp=record.timestamp
    )


@app.post("/v1/score/interval/batch", response_model=BatchResponse)
async def score_interval_batch(req: BatchRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    results: List[BatchResponseItem] = []
    for rec in req.records:
        try:
            resp = await score_interval(rec, x_api_key)
            results.append(BatchResponseItem(timestamp=resp.timestamp, interval_score=resp.interval_score,
                                             model_version=resp.model_version, confidence=resp.confidence))
        except HTTPException as he:
            # skip bad records but continue
            logger.warning(f"Skipping bad record: {he.detail}")
        except Exception:
            logger.exception("Error scoring record; skipping")
    return BatchResponse(results=results, processed=len(results))


@app.post("/v1/score/session", response_model=SessionResponse)
async def score_session(req: SessionRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)

    # compute rule-based aggregate score (simple average of rules per avg features)
    rule_session = rule_score(req.avg_temperature, req.avg_humidity, req.avg_sound_level, req.avg_light)

    residual_session = None
    model_ver = model_service.version
    try:
        # create simple feature vector for session residual model: same 4 features
        X = np.array([[req.avg_temperature, req.avg_humidity, req.avg_light, req.avg_sound_level]])
        if model_service.model is not None:
            pred = model_service.predict_residual(X)
            residual_session = float(pred[0])
        else:
            model_ver = None
    except Exception:
        logger.exception("Session model prediction failed")
        residual_session = None
        model_ver = None

    session_score = float(np.clip(rule_session + (residual_session if residual_session is not None else 0.0), 0.0, 100.0))

    breakdown = {
        "temp_penalty": float(3.5 * abs(req.avg_temperature - TEMP_IDEAL)),
        "hum_penalty": float(0.4 * abs(req.avg_humidity - HUM_IDEAL)),
        "noise_penalty": float(2.0 * max(0.0, req.avg_sound_level - NOISE_THRESHOLD)),
        "light_penalty": float(1.5 * light_penalty_from_lux(req.avg_light, sensor_lower_limit=0.5))
    }

    return SessionResponse(
        session_score=session_score,
        rule_session_score=float(rule_session),
        residual_session=residual_session,
        model_version=model_ver,
        breakdown=breakdown
    )


@app.get("/v1/health")
async def health():
    return {"status": "ok", "model_loaded": model_service.model is not None}


@app.get("/v1/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        model_version=model_service.version,
        trained_at=None,
        artifact_uri=model_service.artifact_path,
        features=["temperature", "humidity", "light", "sound_level"],
        expected_ranges=EXPECTED_RANGES,
        notes="Hybrid = rule + residual; interval = 5 minutes"
    )


@app.post("/v1/model/reload")
async def model_reload(admin_key: Optional[str] = Header(None)):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="admin key required")
    model_service.load_model()
    return {"status": "reloaded", "model_version": model_service.version}


# Dev-only verbose endpoint (protect in production)
@app.post("/v1/debug/score")
async def debug_score(record: IntervalRecord, admin_key: Optional[str] = Header(None)):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="admin key required")
    # return detailed contributions
    r_score = rule_score(record.temperature, record.humidity, record.sound_level, record.light)
    feats = compute_features_from_interval(record)
    residual = None
    try:
        if model_service.model is not None:
            pred = model_service.predict_residual(feats)
            residual = float(pred[0])
    except Exception:
        residual = None
    contributions = {
        "temp_component": 100 - (3.5 * abs(record.temperature - TEMP_IDEAL)),
        "hum_component": 100 - (0.4 * abs(record.humidity - HUM_IDEAL)),
        "noise_component": 100 - (2.0 * max(0.0, record.sound_level - NOISE_THRESHOLD)),
        "light_component": 100 - (1.5 * light_penalty_from_lux(record.light, sensor_lower_limit=0.5))
    }
    return {"rule_score": r_score, "residual": residual, "contributions": contributions}


# -----------------------------
# If run as main (for dev)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_ml_scaffold:app", host="0.0.0.0", port=8000, reload=True)
