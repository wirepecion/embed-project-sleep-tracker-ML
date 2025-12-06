import joblib
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model/model.joblib")

# Global variable to hold the model in RAM
_GLOBAL_MODEL = None

def load_model_into_memory():
    """Call this explicitly on app startup."""
    global _GLOBAL_MODEL
    try:
        if MODEL_PATH.exists():
            _GLOBAL_MODEL = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            # In production, we crash here because the app is useless without the model
            raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise e

def predict_batch(features_list):
    """
    Optimized for batch processing.
    features_list: List of List of floats [[temp, hum...], [temp, hum...]]
    """
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        # Emergency fallback if startup failed, but try not to rely on this
        load_model_into_memory()
    
    # Vectorized prediction is faster than looping
    x = np.array(features_list)
    predictions = _GLOBAL_MODEL.predict(x)
    return predictions.tolist() # Return list of floats