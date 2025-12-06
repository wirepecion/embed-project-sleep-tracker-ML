import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Add project root to path so we can import 'app'
sys.path.append(os.getcwd())

# FIX: Do not import 'db' directly. It causes the NoneType error.
from app.firebase_client import init_firebase
from app.services import process_active_sessions

# --- CONFIGURATION ---
TEST_ID = str(uuid.uuid4())[:8]
SESSION_ID = f"INTEGRATION_TEST_SESSION_{TEST_ID}"
READING_ID = f"INTEGRATION_TEST_READING_{TEST_ID}"

def run_real_test():
    print(f"🚀 STARTING REAL FIREBASE TEST (ID: {TEST_ID})")
    print("------------------------------------------------")

    # 1. Initialize Connection & CAPTURE the client
    # We catch the return value here to ensure we have the live object
    db = init_firebase()
    
    if db is None:
        print("❌ CRITICAL: Could not connect to Firebase. Check Env Vars.")
        return

    print("✅ Firebase Connected.")

    try:
        # 2. SETUP: Create Dummy Data
        print(f"📝 Creating Test Session: {SESSION_ID}")
        db.collection("sleep_sessions").document(SESSION_ID).set({
            "status": "recording",  # <--- Triggers the query
            "started_at": datetime.now(timezone.utc),
            "user_id": "test_bot"
        })

        print(f"📝 Creating Test Reading: {READING_ID}")
        db.collection("sensor_readings").document(READING_ID).set({
            "session_id": SESSION_ID,
            "temperature": 25.5,
            "humidity": 60.0,
            "light": 15.0,
            "sound_level": 40.0,
            "is_processed": False, # <--- Triggers the processing
            "timestamp": datetime.now(timezone.utc)
        })

        # 3. ACT: Run the actual Service Logic
        print("⚙️ Running process_active_sessions() ...")
        
        # This function internally uses the global db, which WAS updated by init_firebase()
        # So inside the app logic, it will work fine.
        process_active_sessions()
        
        print("✅ Service function finished.")

        # 4. VERIFY: Did it happen?
        print("🔍 Verifying results in Database...")
        time.sleep(2) # Give Firestore a moment (eventual consistency)

        # Check A: Reading should be marked processed
        reading_doc = db.collection("sensor_readings").document(READING_ID).get()
        if reading_doc.exists and reading_doc.get("is_processed") is True:
            print("✅ PASS: Sensor Reading marked as 'is_processed=True'")
        else:
            val = reading_doc.get("is_processed") if reading_doc.exists else "DOC_MISSING"
            print(f"❌ FAIL: Sensor Reading is_processed is {val}")

        # Check B: Score should exist
        scores_ref = db.collection("interval_reports").where("session_id", "==", SESSION_ID).stream()
        scores = list(scores_ref)
        
        if len(scores) > 0:
            score_data = scores[0].to_dict()
            print(f"✅ PASS: ML Score found! Value: {score_data.get('score')}")
            print(f"   Document ID: {scores[0].id}")
        else:
            print("❌ FAIL: No ML Score document was created.")

    except Exception as e:
        print(f"❌ ERROR: Test crashed: {e}")

    # finally:
    #     # 5. TEARDOWN: Clean up the mess
    #     print("------------------------------------------------")
    #     print("🧹 Cleaning up test data...")
    #     if db:
    #         try:
    #             db.("sleep_sessions").document(SESSION_ID).delete()
    #             db.("sensor_readings").document(READING_ID).delete()
                
    #             scores_ref = db.("interval_reports").where("session_id", "==", SESSION_ID).stream()
    #             for doc in scores_ref:
    #                 db.("interval_reports").document(doc.id).delete()
                    
    #             print("✅ Cleanup complete. No trace left behind.")
    #         except Exception as e:
    #             print(f"⚠️ Cleanup failed: {e}")

if __name__ == "__main__":
    run_real_test()