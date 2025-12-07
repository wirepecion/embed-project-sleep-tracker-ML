import os
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from google.cloud.firestore import FieldFilter

# Suppress noise
warnings.filterwarnings("ignore", message="Detected filter using positional arguments")

sys.path.append(os.getcwd())

from app.firebase_client import init_firebase
from app.services import process_active_sessions
from app.model_loader import load_model_into_memory 

# --- UPDATED CONFIG ---
TEST_ID = str(uuid.uuid4())[:8]
SESSION_ID = f"INTEGRATION_TEST_SESSION_{TEST_ID}"
READING_ID = f"INTEGRATION_TEST_READING_{TEST_ID}"
TARGET_COLLECTION = "interval_records" # <--- UPDATED to match your new code

def run_real_test():
    print(f"🚀 STARTING TEST (ID: {TEST_ID})")
    print("------------------------------------------------")

    # 1. Load Brains
    load_model_into_memory()
    db = init_firebase()
    
    if db is None:
        print("❌ CRITICAL: No DB Connection.")
        return

    try:
        # 2. SETUP DATA
        print(f"📝 Creating Session: {SESSION_ID}")
        db.collection("sleep_sessions").document(SESSION_ID).set({
            "type": "START",
            "timestamp": datetime.now(timezone.utc),
            "user_id": "test_bot"
        })

        print(f"📝 Creating Reading: {READING_ID}")
        db.collection("sensor_readings").document(READING_ID).set({
            "session_id": SESSION_ID,
            "temperature": 25.5,
            "humidity": 60.0,
            "light": 15.0,
            "sound_level": 40.0,
            "is_processed": False, 
            "timestamp": datetime.now(timezone.utc)
        })

        # 3. ACT (Run the Service)
        print("⚙️ Running process_active_sessions() ...")
        process_active_sessions()
        
        # 4. VERIFY
        print("🔍 Verifying results...")
        time.sleep(2) 

        # Check A: Was reading marked processed?
        reading_doc = db.collection("sensor_readings").document(READING_ID).get()
        if reading_doc.exists and reading_doc.get("is_processed") is True:
            print("✅ PASS: Reading marked 'is_processed'.")
        else:
            print("❌ FAIL: Reading was NOT processed.")

        # Check B: Did we get a report in the NEW collection?
        scores_ref = db.collection(TARGET_COLLECTION)\
            .where(filter=FieldFilter("session_id", "==", SESSION_ID))\
            .stream()
            
        scores = list(scores_ref)
        if len(scores) > 0:
            data = scores[0].to_dict()
            print(f"✅ PASS: Report found in '{TARGET_COLLECTION}'")
            print(f"   Score: {data.get('score'):.2f}")
            print(f"   Base Rule: {data.get('base_rule'):.2f}")
            print(f"   AI Residual: {data.get('ai_residual'):.2f}")
        else:
            print(f"❌ FAIL: No document found in '{TARGET_COLLECTION}'.")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    finally:
        # 5. CLEANUP
        print("------------------------------------------------")
        print("🧹 Cleaning up...")
        # if db:
        #     try:
        #         db.collection("sleep_sessions").document(SESSION_ID).delete()
        #         db.collection("sensor_readings").document(READING_ID).delete()
                
        #         # Delete from new collection
        #         scores_ref = db.collection(TARGET_COLLECTION)\
        #             .where(filter=FieldFilter("session_id", "==", SESSION_ID))\
        #             .stream()
        #         for doc in scores_ref:
        #             db.collection(TARGET_COLLECTION).document(doc.id).delete()
        #     except:
        #         pass
        print("✨ Test Complete.")

if __name__ == "__main__":
    run_real_test()