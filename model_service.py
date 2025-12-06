# model_service.py
import os
import joblib
import numpy as np

MODEL_ARTIFACT = os.environ.get("MODEL_ARTIFACT_PATH", "models/residual_model.joblib")

TEMP_IDEAL = 20.0
HUM_IDEAL = 50.0
NOISE_THRESHOLD = 30.0

def light_penalty(lux):
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
    temp_pen = 3.5 * abs((temp if temp is not None else TEMP_IDEAL) - TEMP_IDEAL)
    hum_pen = 0.4 * abs((hum if hum is not None else HUM_IDEAL) - HUM_IDEAL)
    noise_pen = 2.0 * max(0.0, (noise_db if noise_db is not None else NOISE_THRESHOLD) - NOISE_THRESHOLD)
    light_pen = 1.5 * light_penalty(light if light is not None else 0.0)
    score = 100.0 - (temp_pen + hum_pen + noise_pen + light_pen)
    return float(np.clip(score, 0.0, 100.0))

class ResidualModel:
    def __init__(self, path=MODEL_ARTIFACT):
        self.path = path
        self.model = None
        self.version = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                self.model = joblib.load(self.path)
                self.version = getattr(self.model, "version", os.path.basename(self.path))
                print("Loaded residual model:", self.version)
            except Exception as e:
                print("Failed loading model:", e)
                self.model = None
                self.version = None
        else:
            print("No residual model at", self.path)
            self.model = None
            self.version = None

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Residual model not loaded")
        return self.model.predict(X)

_res = ResidualModel()

def hybrid_predict(temp, hum, noise_db, light):
    r = rule_score(temp, hum, noise_db, light)
    residual = 0.0
    model_version = None
    try:
        if _res.model is not None:
            X = np.array([[temp if temp is not None else TEMP_IDEAL,
                           hum if hum is not None else HUM_IDEAL,
                           light if light is not None else 0.0,
                           noise_db if noise_db is not None else NOISE_THRESHOLD]])
            pred = _res.predict(X)
            residual = float(pred[0])
            model_version = _res.version
    except Exception as e:
        print("Residual predict failed:", e)
        residual = 0.0
        model_version = None
    final = float(np.clip(r + residual, 0.0, 100.0))
    confidence = 0.7 if model_version else 0.45
    return {"interval_score": final, "rule_score": r, "residual": residual, "model_version": model_version, "confidence": confidence}
