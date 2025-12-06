import os
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from google.cloud.firestore import FieldFilter

# 1. CLEANUP: Suppress the specific Google warning if it persists
# warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())

from app.firebase_client import init_firebase
from app.services import process_active_sessions
# 2. FIX: Import the loader so we can prep the model
from app.model_loader import load_model_into_memory 

TEST_ID = str(uuid.uuid4())[:8]
SESSION_ID = f"INTEGRATION_TEST_SESSION_{TEST_ID}"
READING_ID = f"INTEGRATION_TEST_READING_{TEST_ID}"
TARGET_COLLECTION = "interval_reports"

def run_real_test():
    print(f"🚀 STARTING CLEAN TEST (ID: {TEST_ID})")
    print("------------------------------------------------")

    # 3. FIX: Load Model FIRST (Silences 'Attempted prediction with unloaded model')
    load_model_into_memory()

    db = init_firebase()
    if db is None:
        print("❌ CRITICAL: No DB Connection.")
        return

    print("✅ Firebase & Model Ready.")

    try:
        # SETUP
        db.collection("sleep_sessions").document(SESSION_ID).set({
            "type": "START",
            "started_at": datetime.now(timezone.utc),
            "user_id": "test_bot"
        })

        db.collection("sensor_readings").document(READING_ID).set({
            "session_id": SESSION_ID,
            "temperature": 21.9,
            "humidity": 64.20,
            "light": 10.0,
            "sound_level": 48.0,
            "is_processed": False, 
            "timestamp": datetime.now(timezone.utc)
        })

        # ACT
        print("⚙️ Running Service...")
        process_active_sessions()
        
        # VERIFY
        print("🔍 Verifying...")
        time.sleep(2) 

        reading_doc = db.collection("sensor_readings").document(READING_ID).get()
        if reading_doc.exists and reading_doc.get("is_processed") is True:
            print("✅ PASS: Reading processed.")
        else:
            print("❌ FAIL: Reading NOT processed.")

        scores_ref = db.collection(TARGET_COLLECTION)\
            .where(filter=FieldFilter("session_id", "==", SESSION_ID))\
            .stream()
            
        scores = list(scores_ref)
        if len(scores) > 0:
            print(f"✅ PASS: Report Created. Score: {scores[0].to_dict().get('score')}")
        else:
            print(f"❌ FAIL: No Report found.")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    finally:
        # TEARDOWN
        # if db:
        #     try:
        #         db.collection("sleep_sessions").document(SESSION_ID).delete()
        #         db.collection("sensor_readings").document(READING_ID).delete()
        #         scores_ref = db.collection(TARGET_COLLECTION)\
        #             .where(filter=FieldFilter("session_id", "==", SESSION_ID))\
        #             .stream()
        #         for doc in scores_ref:
        #             db.collection(TARGET_COLLECTION).document(doc.id).delete()
        #     except:
        #         pass
        print("------------------------------------------------")
        print("✨ Test Complete.")

if __name__ == "__main__":
    run_real_test()