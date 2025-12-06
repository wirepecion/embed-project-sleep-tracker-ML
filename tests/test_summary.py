import os, sys, time, uuid
from datetime import datetime, timezone
sys.path.append(os.getcwd())
from app.firebase_client import init_firebase
from app.services import process_finished_sessions

TEST_ID = str(uuid.uuid4())[:8]
SESSION_ID = f"TEST_SUMMARY_{TEST_ID}"

def run_test():
    db = init_firebase()
    print(f"🚀 Creating Finished Session: {SESSION_ID}")
    
    # 1. Create Session with type="END"
    db.collection("sleep_sessions").document(SESSION_ID).set({
        "type": "END",
        "timestamp": datetime.now(timezone.utc)
    })
    
    # 2. Add fake readings
    print("📝 Adding fake readings...")
    for i in range(3):
        db.collection("sensor_readings").add({
            "session_id": SESSION_ID,
            "temperature": 25.0, "humidity": 50.0, "light": 0.0, "sound_level": 30.0,
            "timestamp": datetime.now(timezone.utc),
            "is_processed": True
        })
    # Add fake scores
    for i in range(3):
        db.collection("interval_reports").add({
            "session_id": SESSION_ID, "score": 90.0 + i
        })

    # 3. Run Aggregator
    print("⚙️ Running Aggregator...")
    process_finished_sessions()
    
    # 4. Verify
    print("🔍 Checking session_records...")
    time.sleep(2)
    summary = db.collection("session_records").document(SESSION_ID).get()
    
    if summary.exists:
        data = summary.to_dict()
        print(f"✅ SUCCESS! Summary created.")
        print(f"   Avg Temp: {data['averageTemperature']}")
        print(f"   Score: {data['sleepQualityScore']}")
    else:
        print("❌ FAIL: No summary document found.")

    # Cleanup
    # (Add cleanup code here if you want)

if __name__ == "__main__":
    run_test()