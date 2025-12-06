# firebase_client.py
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

_db = None
_initialized = False

def _write_env_json_to_file():
    """
    If FIREBASE_CREDENTIALS_JSON env var exists, write it to FIREBASE_CREDENTIALS_PATH.
    Returns path if written or exists, else None.
    """
    json_env = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "/secrets/serviceAccountKey.json")
    # if file already exists, use it
    if os.path.exists(cred_path):
        return cred_path
    if json_env:
        try:
            os.makedirs(os.path.dirname(cred_path), exist_ok=True)
            with open(cred_path, "w") as f:
                f.write(json_env)
            os.chmod(cred_path, 0o600)
            return cred_path
        except Exception:
            return None
    return None

def init_firebase():
    global _db, _initialized
    if _initialized:
        return _db
    _initialized = True

    # try to ensure a credentials file exists (from env var or mounted file)
    cred_path = _write_env_json_to_file()

    try:
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            _db = firestore.client()
            print("🔥 Firebase initialized with service account.")
            return _db
        # fallback: attempt ADC (only works if running in GCP with appropriate service account)
        try:
            firebase_admin.initialize_app()
            _db = firestore.client()
            print("🔁 Firebase initialized with Application Default Credentials.")
            return _db
        except Exception:
            print("⚠️ Firebase not initialized (no credentials).")
            _db = None
            return None
    except Exception as e:
        print("❌ Firebase init failed:", e)
        _db = None
        return None
