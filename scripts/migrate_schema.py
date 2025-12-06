import firebase_admin
from firebase_admin import credentials, firestore
import os
import base64
import json

# Setup (same as your client)
b64 = os.getenv("FIREBASE_CREDENTIALS_JSON_B64")
if b64:
    creds_dict = json.loads(base64.b64decode(b64).decode("utf-8"))
    creds = credentials.Certificate(creds_dict)
    firebase_admin.initialize_app(creds)
db = firestore.client()

def backfill_is_processed():
    print("🚀 Starting Schema Migration...")
    
    # Get all readings that don't have the 'is_processed' field
    # (We have to scan all because Firestore can't easily query 'missing' fields)
    docs = db.collection("sensor_readings").stream()
    
    batch = db.batch()
    count = 0
    
    for doc in docs:
        data = doc.to_dict()
        
        # If the field is missing, we add it.
        if "is_processed" not in data:
            ref = db.collection("sensor_readings").document(doc.id)
            # Default to False so the ML model picks it up
            batch.update(ref, {"is_processed": False}) 
            count += 1
            
            # Commit every 400 updates (Firestore limit is 500)
            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
                print(f"   Updated {count} docs...")

    if count % 400 != 0:
        batch.commit()

    print(f"✅ Migration Complete. {count} documents updated to include 'is_processed: false'.")

if __name__ == "__main__":
    backfill_is_processed()