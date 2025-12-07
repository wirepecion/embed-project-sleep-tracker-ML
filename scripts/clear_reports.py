import os
import sys
import time

# Fix path so we can import 'app'
sys.path.append(os.getcwd())

from app.firebase_client import init_firebase

# --- CONFIGURATION ---
TARGET_COLLECTION = "interval_records"
# TARGET_COLLECTION = "sensor_readings"
# TARGET_COLLECTION = "sleep_sessions"
# TARGET_COLLECTION = "session_records"
BATCH_SIZE = 400

def clear_collection():
    print(f"🔥 STARTING CLEANUP: {TARGET_COLLECTION}")
    print("------------------------------------------------")

    # 1. Connect
    db = init_firebase()
    if db is None:
        print("❌ CRITICAL: Could not connect to Firebase.")
        return

    # 2. Query ALL documents in the collection
    # Note: For massive collections (1M+ docs), you would need to paginate this.
    # For testing/development, fetching all stream() is fine.
    docs = list(db.collection(TARGET_COLLECTION).stream())
    total_docs = len(docs)
    
    if total_docs == 0:
        print(f"✅ Collection '{TARGET_COLLECTION}' is already empty.")
        return

    print(f"⚠️  Found {total_docs} documents to delete.")
    print("   Waiting 3 seconds... (Ctrl+C to cancel)")
    time.sleep(3)

    # 3. Batch Delete
    batch = db.batch()
    count = 0
    deleted_total = 0

    for doc in docs:
        batch.delete(doc.reference)
        count += 1

        # Commit when batch gets full
        if count >= BATCH_SIZE:
            batch.commit()
            deleted_total += count
            print(f"   🗑️  Deleted batch of {count}...")
            batch = db.batch() # Start new batch
            count = 0

    # Commit any remaining docs
    if count > 0:
        batch.commit()
        deleted_total += count
        print(f"   🗑️  Deleted final batch of {count}...")

    print("------------------------------------------------")
    print(f"✨ SUCCESS: Wiped {deleted_total} documents from '{TARGET_COLLECTION}'.")

if __name__ == "__main__":
    clear_collection()