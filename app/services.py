from datetime import datetime, timezone
import logging
from typing import Any, List, Dict

# FIX 1: Import the MODULE, not the variable. 
# This guarantees we always see the "live" database connection.
import app.firebase_client as fb_client
from app.model_loader import predict_batch

logger = logging.getLogger("SleepService")

# Constants
COLLECTION_SESSIONS = "sleep_sessions"
COLLECTION_READINGS = "sensor_readings"
COLLECTION_SCORES = "ml_scores"

def get_db():
    """Helper to ensure we always get the live DB object."""
    return fb_client.db

def process_active_sessions():
    """
    1. Query ONLY active sessions (saves DB reads).
    2. For each active session, find unprocessed readings.
    3. Predict and save.
    """
    db = get_db()
    
    # Runtime check: If DB is still None, try to initialize it JIT (Just-In-Time)
    if db is None:
        logger.warning("Database was None in service. Attempting reconnect...")
        db = fb_client.init_firebase()

    if db is None:
        logger.error("Database still not connected. Skipping process cycle.")
        return

    try:
        # Note: The 'UserWarning' about positional arguments is harmless noise from the library.
        # It works fine, but standard syntax is changing in future versions.
        active_sessions_ref = db.collection(COLLECTION_SESSIONS)\
            .where("status", "==", "recording")\
            .stream()

        for session in active_sessions_ref:
            process_single_session(session.id)
            
    except Exception as e:
        logger.error(f"Error querying active sessions: {e}")

def process_single_session(session_id: str):
    db = get_db()
    if db is None: return

    try:
        # STEP 2: Find readings without the 'is_processed' flag
        readings_ref = db.collection(COLLECTION_READINGS)\
            .where("session_id", "==", session_id)\
            .where("is_processed", "==", False)\
            .limit(50) 
        
        docs = list(readings_ref.stream())
        
        if not docs:
            return 

        features_batch = []
        doc_ids = []
        
        # STEP 3: Prepare Data
        for doc in docs:
            data = doc.to_dict()
            try:
                feats = [
                    float(data.get("temperature", 0)),
                    float(data.get("humidity", 0)),
                    float(data.get("light", 0)),
                    float(data.get("sound_level", 0))
                ]
                features_batch.append(feats)
                doc_ids.append(doc.id)
            except Exception as e:
                logger.error(f"Corrupt data in {doc.id}: {e}")

        if not features_batch:
            return

        # STEP 4: Batch Predict
        scores = predict_batch(features_batch)
        
        # STEP 5: Write results
        batch = db.batch()
        
        for i, doc_id in enumerate(doc_ids):
            score = scores[i]
            
            score_ref = db.collection(COLLECTION_SCORES).document() 
            score_data = {
                "session_id": session_id,
                "reading_id": doc_id,
                "score": float(score),
                "created_at": datetime.now(timezone.utc)
            }
            batch.set(score_ref, score_data)
            
            reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
            batch.update(reading_ref, {"is_processed": True})

        batch.commit()
        logger.info(f"Processed {len(docs)} readings for session {session_id}")

    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {e}")