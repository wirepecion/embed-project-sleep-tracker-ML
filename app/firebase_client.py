import json
import base64
import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, Any

logger = logging.getLogger(__name__)

# --- GLOBAL SINGLETON DB CLIENT ---
# We define it here so other modules can import 'db'
# Pylance will now see 'db' as an exportable symbol.
db: Optional[firestore.Client] = None 

def decode_credentials():
    b64 = os.getenv("FIREBASE_CREDENTIALS_JSON_B64")
    if not b64:
        logger.error("Missing FIREBASE_CREDENTIALS_JSON_B64 env")
        return None

    try:
        decoded = base64.b64decode(b64).decode("utf-8")
        return json.loads(decoded)
    except Exception as e:
        logger.critical(f"Failed to decode credentials: {e}")
        return None

def init_firebase():
    """
    Initializes the Firebase App and sets the global 'db' client.
    Call this ONCE at startup.
    """
    global db
    
    # If already initialized, just return
    if db is not None:
        return db

    try:
        # Check if app is already initialized in firebase_admin
        if not firebase_admin._apps:
            creds_dict = decode_credentials()
            if not creds_dict:
                raise ValueError("Invalid Credentials")
            
            creds = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(creds)
        
        # Initialize Firestore Client
        db = firestore.client()
        logger.info("Firestore connected successfully.")
        return db
        
    except Exception as e:
        logger.critical(f"Firebase Init Failed: {e}")
        # In production, we might want to kill the app here if DB is critical
        db = None
        return None