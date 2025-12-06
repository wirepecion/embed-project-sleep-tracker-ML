from datetime import datetime, timezone
import logging
import app.firebase_client as fb_client
from app.model_loader import predict_batch
# FIX: Explicit import
from google.cloud.firestore import FieldFilter

logger = logging.getLogger("SleepService")

COLLECTION_SESSIONS = "sleep_sessions"
COLLECTION_READINGS = "sensor_readings"
COLLECTION_SCORES = "interval_reports"

KEY_SESSION_STATUS = "type"      
VAL_SESSION_ACTIVE = "START"     
KEY_PROCESSED = "is_processed"

def get_db():
    return fb_client.db

def process_active_sessions():
    db = get_db()
    if db is None:
        db = fb_client.init_firebase()
    if db is None: return

    try:
        # FIX: Using 'filter=' keyword argument
        active_sessions_ref = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ACTIVE))\
            .stream()

        for session in active_sessions_ref:
            process_single_session(session.id)
            
    except Exception as e:
        logger.error(f"Error querying active sessions: {e}")

def process_single_session(session_id: str):
    db = get_db()
    if db is None: return

    try:
        # FIX: Using 'filter=' keyword argument for BOTH filters
        readings_ref = db.collection(COLLECTION_READINGS)\
            .where(filter=FieldFilter("session_id", "==", session_id))\
            .where(filter=FieldFilter(KEY_PROCESSED, "==", False))\
            .limit(50) 
        
        docs = list(readings_ref.stream())
        
        if not docs:
            return 

        features_batch = []
        doc_ids = []
        
        for doc in docs:
            data = doc.to_dict()
            try:
                feats = [
                    float(data.get("temperature", 0.0)),
                    float(data.get("humidity", 0.0)),
                    float(data.get("light", 0.0)),
                    float(data.get("sound_level", 0.0))
                ]
                features_batch.append(feats)
                doc_ids.append(doc.id)
            except Exception as e:
                logger.error(f"Corrupt data in {doc.id}: {e}")

        if not features_batch:
            return

        scores = predict_batch(features_batch)
        
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
            batch.update(reading_ref, {KEY_PROCESSED: True})

        batch.commit()
        logger.info(f"✅ Processed {len(docs)} readings for session {session_id}")

    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {e}")