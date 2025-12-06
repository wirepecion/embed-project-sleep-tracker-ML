import os
import firebase_admin
from firebase_admin import credentials, firestore

db = None

def init_firebase():
    global db

    if db is not None:
        return db  # already initialized

    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")

    # Case 1: JSON exists → load it
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("🔥 Firebase initialized with service account")
        return db

    # Case 2: No JSON → skip logging (safe fallback)
    print("⚠️ No Firebase credentials found. Firestore logging disabled.")
    db = None
    return db
