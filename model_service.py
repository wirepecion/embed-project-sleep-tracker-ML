# model_service.py
import numpy as np
import joblib
import os

MODEL_PATH = os.environ.get("MODEL_ARTIFACT_PATH", "models/residual_model.joblib")

# rule constants
TEMP_IDEAL = 20.0
HUM_IDEAL = 50.0
NOISE_THRESHOLD = 30.0

def light_penalty(lux: float) -> float:
    if lux is None:
        return 0.0
    if lux <= 1.0:
        return 0.0
    if lux <= 10.0:
        return (lux - 1.0)
    base = (10.0 - 1.0)
    extra = 0.6 * (lux - 10.0)
    return base + extra

def rule_score(temp, hum, noise_db, light):
    temp_pen = 3.5 * abs((temp or TEMP_IDEAL) - TEMP_IDEAL)
    hum_pen = 0.4 * abs((hum or HUM_IDEAL) - HUM_IDEAL)
    noise_pen = 2.0 * max(0.0, (noise_db or NOISE_THRESHOLD) - NOISE_THRESHOLD)
    light_pen = 1.5 * light_penalty(light or 0.0)
    score = 100.0 - (temp_pen + hum_pen + noise_pen + light_pen)
    return float(np.clip(score, 0.0, 100.0))

# Model wrapper
class ResidualModel:
    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.model = None
        self.version = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                self.model = joblib.load(self.path)
                self.version = getattr(self.model, "version", os.path.basename(self.path))
                print("Loaded model:", self.version)
            except Exception as e:
                print("Failed to load model:", e)
                self.model = None
                self.version = None
        else:
            print("Model artifact not found at", self.path)
            self.model = None
            self.version = None

    def predict(self, feature_array):
        if self.model is None:
            raise RuntimeError("No model loaded")
        return self.model.predict(feature_array)

# create singleton
res_model = ResidualModel()

def hybrid_predict(temp, hum, noise_db, light):
    r = rule_score(temp, hum, noise_db, light)
    residual = 0.0
    model_version = None
    try:
        if res_model.model is not None:
            X = np.array([[temp or TEMP_IDEAL, hum or HUM_IDEAL, (light or 0.0), (noise_db or NOISE_THRESHOLD)]])
            pred = res_model.predict(X)
            residual = float(pred[0])
            model_version = res_model.version
    except Exception as e:
        print("Residual prediction failed:", e)
        residual = 0.0
        model_version = None

    hybrid = float(np.clip(r + residual, 0.0, 100.0))
    # simple confidence: lower if model missing
    confidence = 0.7 if model_version else 0.4
    return {"interval_score": hybrid, "rule_score": r, "residual": residual, "model_version": model_version, "confidence": confidence}
