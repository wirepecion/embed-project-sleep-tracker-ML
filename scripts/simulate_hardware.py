import os
import sys
import time
import random
import uuid
from datetime import datetime, timezone, timedelta

# Fix path to allow importing from 'app'
sys.path.append(os.getcwd())

from app.firebase_client import init_firebase

# --- CONFIGURATION ---
SIMULATION_SPEED_SECONDS = 2  # Wait 2 real seconds between readings
TOTAL_READINGS_TO_SEND = 1   # How many "5-minute" packets to send (5 packets = 25 virtual minutes)
SESSION_ID = f"SIM_SESSION_{str(uuid.uuid4())[:6]}"

def simulate_night_sleep():
    print(f"🤖 HARDWARE SIMULATOR STARTING")
    print(f"   Target Session ID: {SESSION_ID}")
    print("------------------------------------------------")

    # 1. Connect
    db = init_firebase()
    if db is None:
        print("❌ CRITICAL: No DB Connection.")
        return

    # 2. START SESSION
    print(f"🟢 [1/3] Sending START Signal...")
    start_payload = {
        "type": "START",
        "timestamp": datetime.now(timezone.utc),  # <--- START TIME
        # "user_id": "simulated_user_01",
        # "device_id": "esp32_prototype_A"
    }
    db.collection("sleep_sessions").document(SESSION_ID).set(start_payload)
    print("   Session Active. The server should detect this within 30s.")

    # 3. SENSOR LOOP
    print(f"🔄 [2/3] Sending {TOTAL_READINGS_TO_SEND} sensor readings...")
    
    # Starting values (Typical Thai Night)
    current_temp = 26.5
    current_hum = 60.0
    current_noise = 35.0
    current_light = 130.0 # Dark room

    # Virtual clock starts now
    virtual_time = datetime.now(timezone.utc)

    for i in range(TOTAL_READINGS_TO_SEND):
        # Drift the values slightly (Random Walk) to look real
        current_temp += random.uniform(-0.1, 0.1)
        current_hum += random.uniform(-0.5, 0.5)
        current_noise = max(30, current_noise + random.uniform(-2, 2)) # Occasional noise
        
        # Send Data
        payload = {
            "session_id": SESSION_ID,
            "temperature": round(current_temp, 2),
            "humidity": round(current_hum, 2),
            "light": round(current_light, 1),
            "sound_level": round(current_noise, 1),
            "timestamp": virtual_time,
            "is_processed": False # <--- CRITICAL: This triggers your ML Service
        }
        
        # Write to DB
        db.collection("sensor_readings").add(payload)
        
        print(f"   📡 Sent Reading #{i+1} (Virtual Time: {virtual_time.strftime('%H:%M')})")
        print(f"      Temp: {current_temp:.1f}C | Noise: {current_noise:.1f}dB")

        # Advance virtual time by 5 minutes, wait real time by 2 seconds
        virtual_time += timedelta(minutes=5)
        time.sleep(SIMULATION_SPEED_SECONDS)

    # 4. END SESSION
    print(f"🔴 [3/3] Sending END Signal...")
    end_payload = {
    "type": "END",
    "ended_at": datetime.now(timezone.utc)    # <--- END TIME
    }
    db.collection("sleep_sessions").document(SESSION_ID).update(end_payload)
    
    print("------------------------------------------------")
    print("✨ SIMULATION COMPLETE")
    print(f"   Check your dashboard for Session ID: {SESSION_ID}")
    print("   Your 'Session Aggregator' should pick this up in ~30 seconds.")

if __name__ == "__main__":
    simulate_night_sleep()