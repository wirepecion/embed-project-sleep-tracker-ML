import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("model/model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict(features):
    model = load_model()
    x = np.array(features).reshape(1, -1)
    y = model.predict(x)[0]
    return float(y)
