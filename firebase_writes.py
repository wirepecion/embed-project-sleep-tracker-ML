# firebase_writes.py
from firebase_admin import firestore
from typing import Optional

def env_doc_path(session_id: str, iso_ts: str):
    return f"sessions/{session_id}/environment/{iso_ts}"

def ml_doc_path(session_id: str, iso_ts: str):
    return f"sessions/{session_id}/ml_scores/{iso_ts}"

def write_environment_record(db, session_id: str, iso_ts: str, payload: dict) -> bool:
    """
    db: firestore client or None
    payload: basic sensor payload (temp, hum, light, noise) - already validated
    """
    if db is None:
        return False, None
    coll = db.collection("sessions").document(session_id).collection("environment")
    doc = coll.document(iso_ts)
    doc.set(payload)
    return True, f"sessions/{session_id}/environment/{iso_ts}"

def write_ml_score(db, session_id: str, iso_ts: str, payload: dict) -> bool:
    if db is None:
        return False, None
    coll = db.collection("sessions").document(session_id).collection("ml_scores")
    doc = coll.document(iso_ts)
    doc.set(payload)
    return True, f"sessions/{session_id}/ml_scores/{iso_ts}"
