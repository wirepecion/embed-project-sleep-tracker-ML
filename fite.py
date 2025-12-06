from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import os

# -----------------------------
# Firebase Initialization
# -----------------------------
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
        print("Warning: Firebase credentials not found. Firestore disabled.")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("/home/sirav/sleep_tracker_ML/models/residual_model.joblib")  # must contain: temperature, humidity, noise, light


# -----------------------------
# Request / Response Schemas
# -----------------------------

class EnvironmentInput(BaseModel):
    timestamp: str       # Example: "December 5, 2025 at 2:52:08 PM UTC+7"
    temperature: float
    humidity: float
    noise: float
    light: float


class PredictionResponse(BaseModel):
    timestamp: str
    predicted_score: float
    segment: Optional[str]


# -----------------------------
# Helper: Firestore Save
# -----------------------------
def save_to_firestore(collection: str, doc_id: str, data: dict):
    if db:
        db.collection(collection).document(doc_id).set(data)


# -----------------------------
# Helper: Convert timestamp to document id
# -----------------------------
def normalize_ts(ts: str) -> str:
    # Firestore-safe ID
    return ts.replace(" ", "_").replace(":", "-")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Sleep Environment ML API")


# -----------------------------
# Endpoint: Predict per 5 minutes
# -----------------------------
@app.post("/predict-environment", response_model=PredictionResponse)
def predict_environment(payload: EnvironmentInput):

    # Prepare features
    features = [[
        payload.temperature,
        payload.humidity,
        payload.noise,
        payload.light
    ]]

    # ML prediction
    score = float(model.predict(features)[0])

    # Classification
    if score >= 80:
        seg = "excellent"
    elif score >= 60:
        seg = "good"
    elif score >= 40:
        seg = "fair"
    else:
        seg = "poor"

    # Save raw environmental data
    doc_id = normalize_ts(payload.timestamp)
    save_to_firestore(
        "environment_logs",
        doc_id,
        {
            "timestamp": payload.timestamp,
            "temperature": payload.temperature,
            "humidity": payload.humidity,
            "noise": payload.noise,
            "light": payload.light
        }
    )

    # Save prediction
    save_to_firestore(
        "environment_predictions",
        doc_id,
        {
            "timestamp": payload.timestamp,
            "predicted_score": score,
            "segment": seg
        }
    )

    return PredictionResponse(
        timestamp=payload.timestamp,
        predicted_score=score,
        segment=seg
    )


# -----------------------------
# OPTIONAL: Predict existing Firestore history at startup
# -----------------------------
@app.on_event("startup")
def run_startup_prediction():
    if not db:
        return

    logs_ref = db.collection("environment_logs").stream()

    for doc in logs_ref:
        data = doc.to_dict()
        ts = data["timestamp"]
        doc_id = normalize_ts(ts)

        # Skip if prediction already exists
        pred_ref = db.collection("environment_predictions").document(doc_id).get()
        if pred_ref.exists:
            continue

        # Run model
        features = [[
            data["temperature"],
            data["humidity"],
            data["noise"],
            data["light"]
        ]]

        score = float(model.predict(features)[0])
        seg = "excellent" if score >= 80 else "good" if score >= 60 else "fair" if score >= 40 else "poor"

        # Save prediction
        save_to_firestore(
            "environment_predictions",
            doc_id,
            {
                "timestamp": ts,
                "predicted_score": score,
                "segment": seg
            }
        )
