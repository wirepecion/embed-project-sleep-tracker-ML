from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import joblib
import os

# Firebase wrapper
from firebase_client import init_firebase
import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI

# Initialize Firebase
try:
    firebase_admin.get_app()
except ValueError:
    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        db = None
        print("⚠️ Firebase credentials NOT found. Firestore disabled.")



# ---------------------------
# Pydantic Models
# ---------------------------

class IntervalInput(BaseModel):
    timestamp: str
    temperature_c: float
    humidity_percent: float
    noise_db: float
    light_lux: float

class IntervalResponse(BaseModel):
    comfort_score: float
    model_version: str
    wrote_to_firestore: bool


class SessionEndRequest(BaseModel):
    session_start_ts: str
    session_end_ts: str


class SessionEndResponse(BaseModel):
    avg_score: float
    total_intervals: int
    wrote_to_firestore: bool


# ---------------------------
# ML Model Loader
# ---------------------------

MODEL_PATH = "models/residual_model.joblib"
MODEL_VERSION = "v1.0.0"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("🧠 ML model loaded:", MODEL_PATH)
    else:
        print("⚠️ No model found. Using fallback rule-based scoring.")
        model = None


# ---------------------------
# Rule-Based Backup Model
# ---------------------------

def fallback_rule_score(payload: IntervalInput) -> float:
    score = 100.0
    score -= abs(payload.temperature_c - 24.0) * 3
    score -= abs(payload.humidity_percent - 55) * 0.5
    score -= max(0, payload.noise_db - 30) * 1.2
    score -= max(0, payload.light_lux - 15) * 0.8
    return max(0, min(100, score))


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(title="Sleep Comfort ML API")


@app.on_event("startup")
def startup_event():
    print("🚀 Starting ML API...")
    load_model()
    init_firebase()   # Firebase loads safely **here**, not during import


# ---------------------------
# Firestore write helper
# ---------------------------

def write_interval_to_firestore(data: dict):
    db = init_firebase()
    if db is None:
        return False

    ref = db.collection("sleep_data")
    ref.add(data)
    return True


# ---------------------------
# API: Predict Interval
# ---------------------------

@app.post("/v1/score/interval", response_model=IntervalResponse)
def score_interval(payload: IntervalInput):

    # Use ML model if available
    if model:
        features = [[
            payload.temperature_c,
            payload.humidity_percent,
            payload.noise_db,
            payload.light_lux
        ]]
        comfort_score = float(model.predict(features)[0])
    else:
        comfort_score = fallback_rule_score(payload)

    fs_data = {
        "timestamp": payload.timestamp,
        "temperature_c": payload.temperature_c,
        "humidity_percent": payload.humidity_percent,
        "noise_db": payload.noise_db,
        "light_lux": payload.light_lux,
        "comfort_score": comfort_score,
        "model_version": MODEL_VERSION
    }

    wrote = write_interval_to_firestore(fs_data)

    return IntervalResponse(
        comfort_score=comfort_score,
        model_version=MODEL_VERSION,
        wrote_to_firestore=wrote
    )


# ---------------------------
# API: End Session → Summary
# ---------------------------

@app.post("/v1/session/end", response_model=SessionEndResponse)
def end_session(payload: SessionEndRequest):
    db = init_firebase()
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore not configured")

    start_ts = payload.session_start_ts
    end_ts = payload.session_end_ts

    ref = db.collection("sleep_data")
    docs = ref.where("timestamp", ">=", start_ts).where("timestamp", "<=", end_ts).stream()

    scores = []
    for d in docs:
        x = d.to_dict()
        if "comfort_score" in x:
            scores.append(x["comfort_score"])

    if len(scores) == 0:
        raise HTTPException(status_code=404, detail="No interval scores in that range")

    avg_score = sum(scores) / len(scores)

    summary = {
        "session_start_ts": start_ts,
        "session_end_ts": end_ts,
        "avg_score": avg_score,
        "total_intervals": len(scores),
    }

    # Write to Firestore
    db.collection("sleep_sessions").add(summary)

    return SessionEndResponse(
        avg_score=avg_score,
        total_intervals=len(scores),
        wrote_to_firestore=True
    )


# ---------------------------
# Health
# ---------------------------

@app.get("/v1/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------
@app.get("/test-firestore")
def test_firestore():
    if db is None:
        return {"status": "error", "message": "Firestore is NOT initialized"}

    test_ref = db.collection("test_collection").document("connectivity_test")

    # Write sample data
    test_ref.set({
        "message": "Hello Firestore!",
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # Read back
    doc = test_ref.get()
    if doc.exists:
        return {
            "status": "success",
            "written_data": doc.to_dict()
        }

    return {"status": "error", "message": "Document write failed"}