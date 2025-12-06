from datetime import datetime, timezone
from app.firebase_client import db # Assuming your init code is here
from app.model_loader import predict_batch
import logging

logger = logging.getLogger("SleepService")

# Constants
COLLECTION_SESSIONS = "sleep_sessions"
COLLECTION_READINGS = "sensor_readings"
COLLECTION_SCORES = "ml_scores"

def process_active_sessions():
    """
    1. Query ONLY active sessions (saves DB reads).
    2. For each active session, find unprocessed readings.
    3. Predict and save.
    """
    if db is None:
        logger.warning("Database not connected.")
        return

    # STEP 1: Optimization - Only get sessions that are currently recording
    # You MUST create a Composite Index in Firebase Console for this query: 'status' + 'last_active'
    active_sessions_ref = db.collection(COLLECTION_SESSIONS)\
        .where("status", "==", "recording")\
        .stream()

    for session in active_sessions_ref:
        process_single_session(session.id)

def process_single_session(session_id: str):
    """
    Fetch readings that don't have a score yet.
    """
    # STEP 2: Find readings without the 'is_processed' flag
    # Note: Using a flag in the reading doc is safer than checking timestamps
    readings_ref = db.collection(COLLECTION_READINGS)\
        .where("session_id", "==", session_id)\
        .where("is_processed", "==", False)\
        .limit(50) # Batch size limit to prevent memory overflow
    
    docs = list(readings_ref.stream())
    
    if not docs:
        return # Nothing new to process

    features_batch = []
    doc_ids = []
    
    # STEP 3: Prepare Data
    for doc in docs:
        data = doc.to_dict()
        # Ensure feature order MATCHES your model training exactly!
        # Example: [temp, humidity, light, sound]
        try:
            feats = [
                data.get("temperature", 0),
                data.get("humidity", 0),
                data.get("light", 0),
                data.get("sound_level", 0)
            ]
            features_batch.append(feats)
            doc_ids.append(doc.id)
        except Exception as e:
            logger.error(f"Corrupt data in {doc.id}: {e}")

    if not features_batch:
        return

    # STEP 4: Batch Predict (Much faster than 1-by-1)
    try:
        scores = predict_batch(features_batch)
        
        # STEP 5: Write results (Use Batch Writes for Atomicity)
        batch = db.batch()
        
        for i, doc_id in enumerate(doc_ids):
            score = scores[i]
            
            # Ref for the new score document
            score_ref = db.collection(COLLECTION_SCORES).document() 
            
            # Payload
            score_data = {
                "session_id": session_id,
                "reading_id": doc_id,
                "score": float(score),
                "created_at": datetime.now(timezone.utc)
            }
            batch.set(score_ref, score_data)
            
            # Mark reading as processed so we don't read it again
            reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
            batch.update(reading_ref, {"is_processed": True})

        batch.commit()
        logger.info(f"Processed {len(docs)} readings for session {session_id}")

    except Exception as e:
        logger.error(f"Prediction batch failed for {session_id}: {e}")