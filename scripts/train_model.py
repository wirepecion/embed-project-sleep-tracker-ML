# scripts/train_model.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os, json, math, sys

# Fix path to allow importing from 'app'
sys.path.append(os.getcwd())

# IMPORT SHARED TRUTH
from app.sleep_rules import compute_rule_score

# reproducibility
rng = np.random.default_rng(12345)
np.random.seed(12345)

# ------------------ CONFIG ------------------
NUM_NIGHTS = 120
MIN_HOURS_PER_NIGHT = 6
MAX_HOURS_PER_NIGHT = 9

# probabilities and event config (Thai-tuned)
P_HOT_NIGHT = 0.25
P_COLD_NIGHT = 0.03
P_HIGH_HUMIDITY = 0.25
P_LIGHT_SPIKE_PER_HOUR = 0.18
P_NOISE_SPIKE_PER_HOUR = 0.18
AR_RHO = 0.6

# light profile probs (tuned)
P_BLACKOUT = 0.08
P_NORMAL = 0.45
P_STREET = 0.22
P_BRIGHT_ROOM = 0.12
P_ALWAYS_LAMP = 0.13

# ------------------ helper functions ------------------
def sample_night_mode(rng):
    r = rng.random()
    if r < P_HOT_NIGHT: return "hot"
    if r < P_HOT_NIGHT + P_COLD_NIGHT: return "cold"
    return "normal"

mode_params = {
    "normal": { "temp_mu": 25.0, "temp_sigma": 1.2, "hum_mu": 65.0, "hum_sigma": 6.0, "noise_mu": 35.0,"noise_sigma": 4.0 },
    "hot":    { "temp_mu": 29.0, "temp_sigma": 1.6, "hum_mu": 72.0, "hum_sigma": 6.0, "noise_mu": 36.0,"noise_sigma": 4.5 },
    "cold":   { "temp_mu": 23.0, "temp_sigma": 0.9, "hum_mu": 60.0, "hum_sigma": 4.0, "noise_mu": 33.0,"noise_sigma": 3.5 }
}

def sample_light_profile(rng):
    r = rng.random()
    if r < P_BLACKOUT: return "blackout"
    if r < P_BLACKOUT + P_NORMAL: return "normal"
    if r < P_BLACKOUT + P_NORMAL + P_STREET: return "street"
    if r < P_BLACKOUT + P_NORMAL + P_STREET + P_BRIGHT_ROOM: return "bright_room"
    return "always_lamp"

def sample_light_base_for_profile(profile, rng):
    if profile == "blackout": return max(0.0, rng.normal(0.05, 0.05))
    if profile == "normal": return max(0.0, rng.normal(2.5, 1.4))
    if profile == "street": return max(0.0, rng.normal(6.0, 3.2))
    if profile == "bright_room": return max(0.0, rng.normal(80.0, 25.0))
    if profile == "always_lamp": return max(0.0, rng.normal(10.0, 4.0))
    return max(0.0, rng.normal(2.0, 1.0))

# ------------------ generator ------------------
def generate_one_night(night_idx, rng):
    base_date = datetime(2025, 1, 1) + timedelta(days=night_idx)
    start_hour = int(rng.integers(22, 25))
    start_time = base_date.replace(hour=0, minute=0, second=0) + timedelta(hours=start_hour)
    hours = int(rng.integers(MIN_HOURS_PER_NIGHT, MAX_HOURS_PER_NIGHT + 1))
    timestamps = [start_time + timedelta(hours=h) for h in range(hours)]

    mode = sample_night_mode(rng)
    params = mode_params[mode].copy()

    if rng.random() < P_HIGH_HUMIDITY: params["hum_mu"] += 6.0

    temp_prev = float(rng.normal(params["temp_mu"], params["temp_sigma"]))
    hum_prev = float(rng.normal(params["hum_mu"], params["hum_sigma"]))
    noise_prev = float(rng.normal(params["noise_mu"], params["noise_sigma"]))

    profile = sample_light_profile(rng)
    light_prev = float(sample_light_base_for_profile(profile, rng))

    rows = []
    session_id = f"session_{night_idx:04d}"

    for i, ts in enumerate(timestamps):
        temp = AR_RHO * temp_prev + (1-AR_RHO) * float(rng.normal(params["temp_mu"], params["temp_sigma"]))
        hum = AR_RHO * hum_prev + (1-AR_RHO) * float(rng.normal(params["hum_mu"], params["hum_sigma"]))
        noise = AR_RHO * noise_prev + (1-AR_RHO) * float(rng.normal(params["noise_mu"], params["noise_sigma"]))
        light = AR_RHO * light_prev + (1-AR_RHO) * float(sample_light_base_for_profile(profile, rng))

        if rng.random() < P_LIGHT_SPIKE_PER_HOUR:
            if profile == "bright_room": light += abs(rng.normal(25.0, 12.0))
            elif profile == "always_lamp": light += abs(rng.normal(10.0, 5.0))
            else: light += abs(rng.normal(8.0, 4.0))

        if rng.random() < P_NOISE_SPIKE_PER_HOUR:
            noise += abs(rng.normal(14.0, 8.0))

        temp = float(np.clip(temp + float(rng.normal(0,0.2)), 15.0, 40.0))
        hum = float(np.clip(hum + float(rng.normal(0,0.8)), 20.0, 100.0))
        noise = float(np.clip(noise + float(rng.normal(0,0.8)), 15.0, 120.0))
        light = float(np.clip(light + float(rng.normal(0,0.4)), 0.0, 2000.0))

        # USE SHARED RULE
        rule_score = compute_rule_score(temp, hum, noise, light)

        # The "Real" score has some subjective noise added
        observed_score = float(np.clip(rule_score + rng.normal(0.0, 4.0), 0.0, 100.0))

        rows.append({
            "session_id": session_id,
            "timestamp": ts,
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "light": round(light, 2),
            "noise_db": round(noise, 2),
            "profile": profile,
            "rule_score_deterministic": round(rule_score, 2),
            "comfort_score": round(observed_score, 2)
        })

        temp_prev, hum_prev, noise_prev, light_prev = temp, hum, noise, light

    return rows

if __name__ == "__main__":
    print("🚀 Generating Thai Sleep Data...")
    
    # generate dataset
    all_rows = []
    for night in range(NUM_NIGHTS):
        all_rows.extend(generate_one_night(night, rng))

    df = pd.DataFrame(all_rows).sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    df["residual"] = df["comfort_score"] - df["rule_score_deterministic"]

    # save csv
    os.makedirs("data", exist_ok=True)
    csv_path = "data/mock_sleep_thai_profiles.csv"
    df.to_csv(csv_path, index=False)

    # ------------------ train residual model ------------------
    feature_cols = ["temperature", "humidity", "noise_db", "light"]
    X = df[feature_cols].values
    y_res = df["residual"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_res, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print("🧠 Training Random Forest Residual Model...")
    rf_res = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_res.fit(X_train, y_train)

    # save model and metrics
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.joblib" # PRODUCTION PATH
    joblib.dump(rf_res, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Data saved to: {csv_path}")