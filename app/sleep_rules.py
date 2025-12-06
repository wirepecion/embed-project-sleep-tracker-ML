import numpy as np

# ------------------ CONFIG ------------------
TEMP_IDEAL = 25.5   # °C
HUM_IDEAL = 60.0    # %RH
NOISE_THRESHOLD = 45.0  # dB
SENSOR_LOWER_LIMIT = 0.5

def light_penalty_from_lux(lux, sensor_lower_limit=None):
    if lux is None: return None
    if sensor_lower_limit is not None and lux == 0:
        lux = sensor_lower_limit
    if lux <= 1.0: return 0.0
    if lux <= 10.0: return 1.0 * (lux - 1.0)
    base = 1.0 * (10.0 - 1.0)
    extra = 0.6 * (lux - 10.0)
    return base + extra

def compute_rule_score(temp, hum, noise_db, light, 
                       sensor_lower_limit=SENSOR_LOWER_LIMIT, 
                       light_sensor_present=True):
    """
    Deterministic physics-based scoring.
    Used by BOTH the Training Script and the Production API.
    """
    score = 100.0
    
    # Temperature Penalty
    if temp is not None:
        score -= 3.5 * abs(temp - TEMP_IDEAL)
        
    # Humidity Penalty
    if hum is not None:
        score -= 0.4 * abs(hum - HUM_IDEAL)
        
    # Noise Penalty
    if noise_db is not None:
        score -= 2.0 * max(0.0, noise_db - NOISE_THRESHOLD)
        
    # Light Penalty
    light_pen = 0.0
    if light_sensor_present:
        lp = light_penalty_from_lux(light, sensor_lower_limit=sensor_lower_limit)
        if lp is not None:
            light_pen = 1.5 * lp
    else:
        # Impute missing light sensor logic
        imputed = 5.0
        lp = light_penalty_from_lux(imputed)
        light_pen = (1.5 * 0.5) * lp
        
    score -= light_pen
    
    return float(np.clip(score, 0.0, 100.0))