# firebase_init.py
import os
import firebase_admin
from firebase_admin import credentials, firestore

def init_firebase():
    """
    Initialize firebase-admin only once. Returns Firestore client or None if credentials missing.
    """
    try:
        firebase_admin.get_app()
    except ValueError:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            print("Firebase initialized from", cred_path)
        else:
            print("Warning: Firebase credentials not found at", cred_path)
            return None
    return firestore.client()

# usage
# from firebase_init import init_firebase
# db = init_firebase()
