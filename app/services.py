from datetime import datetime, timezone
import logging
import app.firebase_client as fb_client
from app.model_loader import predict_batch
from google.cloud.firestore import FieldFilter

# IMPORT YOUR PHYSICS RULES
from app.sleep_rules import compute_rule_score

logger = logging.getLogger("SleepService")

# --- CONFIGURATION ---
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
        # Query for ACTIVE sessions only
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
        # Get unprocessed readings for this session
        readings_ref = db.collection(COLLECTION_READINGS)\
            .where(filter=FieldFilter("session_id", "==", session_id))\
            .where(filter=FieldFilter(KEY_PROCESSED, "==", False))\
            .limit(50) 
        
        docs = list(readings_ref.stream())
        if not docs: return 

        features_batch = []
        doc_ids = []
        rule_scores = []
        
        for doc in docs:
            data = doc.to_dict()
            try:
                # 1. Extract Raw Data
                t = float(data.get("temperature", 0.0))
                h = float(data.get("humidity", 0.0))
                l = float(data.get("light", 0.0))
                n = float(data.get("sound_level", 0.0))
                
                # 2. Compute Physics Score (The Baseline)
                # This ensures we have a base value (e.g., 85.0)
                base_score = compute_rule_score(t, h, n, l)
                rule_scores.append(base_score)

                # 3. Prepare Features for AI
                # Must match training order: [temp, hum, noise, light]
                feats = [t, h, n, l]
                features_batch.append(feats)
                doc_ids.append(doc.id)
            except Exception as e:
                logger.error(f"Corrupt data in {doc.id}: {e}")

        if not features_batch: return

        # 4. Get AI Prediction (The Residual, e.g., -2.5)
        residuals = predict_batch(features_batch)
        
        # 5. Save Results
        batch = db.batch()
        
        for i, doc_id in enumerate(doc_ids):
            base = rule_scores[i]
            resid = residuals[i]
            
            # HYBRID LOGIC: Final = Base + Residual
            final_score = base + resid
            
            # Safety Clip (0 to 100)
            final_score = max(0.0, min(100.0, final_score))
            
            # Write to 'interval_reports'
            score_ref = db.collection(COLLECTION_SCORES).document() 
            score_data = {
                "session_id": session_id,
                "reading_id": doc_id,
                "score": float(final_score),
                "base_rule_score": float(base), # Optional: helps debugging
                "ai_residual": float(resid),    # Optional: helps debugging
                "created_at": datetime.now(timezone.utc)
            }
            batch.set(score_ref, score_data)
            
            # Mark as processed
            reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
            batch.update(reading_ref, {KEY_PROCESSED: True})

        batch.commit()
        logger.info(f"✅ Processed {len(docs)} readings for session {session_id}")

    except Exception as e:
        logger.error(f"Processing failed for session {session_id}: {e}")