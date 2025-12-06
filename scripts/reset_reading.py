import os
import sys
import time

# Fix path so we can import 'app'
sys.path.append(os.getcwd())

from app.firebase_client import init_firebase

# --- CONFIGURATION ---
TARGET_COLLECTION = "sensor_readings"
FIELD_TO_RESET = "is_processed"
RESET_VALUE = False
BATCH_SIZE = 400

def reset_sensor_readings():
    print(f"🔄 STARTING RESET: {TARGET_COLLECTION}")
    print(f"   Setting '{FIELD_TO_RESET}' to {RESET_VALUE} for ALL documents.")
    print("------------------------------------------------")

    # 1. Connect
    db = init_firebase()
    if db is None:
        print("❌ CRITICAL: Could not connect to Firebase.")
        return

    # 2. Query ALL documents
    # Ideally, you'd filter for is_processed=True, but reading everything ensures 
    # we catch any malformed docs too.
    docs = list(db.collection(TARGET_COLLECTION).stream())
    total_docs = len(docs)
    
    if total_docs == 0:
        print(f"⚠️  Collection '{TARGET_COLLECTION}' is empty.")
        return

    print(f"🎯 Found {total_docs} documents to update.")
    print("   Waiting 3 seconds... (Ctrl+C to cancel)")
    time.sleep(3)

    # 3. Batch Update
    batch = db.batch()
    count = 0
    updated_total = 0

    for doc in docs:
        # Optimization: Don't write if it's already False (saves money)
        current_val = doc.to_dict().get(FIELD_TO_RESET)
        if current_val == RESET_VALUE:
            continue

        ref = db.collection(TARGET_COLLECTION).document(doc.id)
        batch.update(ref, {FIELD_TO_RESET: RESET_VALUE})
        count += 1

        # Commit when batch gets full
        if count >= BATCH_SIZE:
            batch.commit()
            updated_total += count
            print(f"   💾 Updated batch of {count}...")
            batch = db.batch() # Start new batch
            count = 0

    # Commit remaining
    if count > 0:
        batch.commit()
        updated_total += count