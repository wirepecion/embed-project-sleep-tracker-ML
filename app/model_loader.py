import joblib
import numpy as np
from pathlib import Path
import logging
from typing import List, Any, Optional

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model/model.joblib")

# --- GLOBAL SINGLETON MODEL ---
# Type hint as Any because joblib can load anything (sklearn, xgboost, etc.)
_GLOBAL_MODEL: Any = None

def load_model_into_memory():
    """Call this explicitly on app startup."""
    global _GLOBAL_MODEL
    try:
        if MODEL_PATH.exists():
            _GLOBAL_MODEL = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            # We don't raise error here to allow app to start, 
            # but predictions will fail.
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        _GLOBAL_MODEL = None

def predict_batch(features_list: List[List[float]]) -> List[float]:
    """
    Optimized for batch processing.
    features_list: List of List of floats [[temp, hum...], [temp, hum...]]
    """
    global _GLOBAL_MODEL
    
    if _GLOBAL_MODEL is None:
        logger.warning("Attempted prediction with unloaded model. Retrying load...")
        load_model_into_memory()
        
    # Final safety check
    if _GLOBAL_MODEL is None:
        logger.error("Model is still None. Returning zeros.")
        return [0.0] * len(features_list)
    
    try:
        # Vectorized prediction is faster than looping
        x = np.array(features_list)
        
        # Pylance fix: We know _GLOBAL_MODEL is not None here, 
        # and we assume it adheres to sklearn interface (.predict)
        predictions = _GLOBAL_MODEL.predict(x)
        
        return predictions.tolist() # Return list of floats
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return [0.0] * len(features_list)