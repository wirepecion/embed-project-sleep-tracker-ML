import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import os

def decode_credentials():
    b64 = os.getenv("FIREBASE_CREDENTIALS_JSON_B64")
    if not b64:
        raise ValueError("Missing FIREBASE_CREDENTIALS_JSON_B64 env")

    decoded = base64.b64decode(b64).decode("utf-8")
    return json.loads(decoded)

def init_firebase():
    if firebase_admin._apps:
        return firebase_admin.get_app()

    creds_dict = decode_credentials()
    creds = credentials.Certificate(creds_dict)
    return firebase_admin.initialize_app(creds)
