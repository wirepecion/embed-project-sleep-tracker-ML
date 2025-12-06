from datetime import datetime, timezone
import logging
import numpy as np
from google.cloud.firestore import FieldFilter

import app.firebase_client as fb_client
from app.model_loader import predict_batch
from app.sleep_rules import compute_rule_score

logger = logging.getLogger("SleepService")

# --- CONFIGURATION ---
COLLECTION_SESSIONS = "sleep_sessions"
COLLECTION_READINGS = "sensor_readings"
COLLECTION_SCORES = "interval_reports"
COLLECTION_SUMMARY = "session_records" # <--- NEW TARGET

KEY_SESSION_STATUS = "type"      
VAL_SESSION_ACTIVE = "START"
VAL_SESSION_ENDED = "END"     
KEY_PROCESSED = "is_processed"

def get_db():
    return fb_client.db

# ---------------------------------------------------------
# 1. REAL-TIME LOOP (Unchanged logic, just cleaner)
# ---------------------------------------------------------
def process_active_sessions():
    db = get_db()
    if db is None: db = fb_client.init_firebase()
    if db is None: return

    try:
        active_sessions_ref = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ACTIVE))\
            .stream()

        for session in active_sessions_ref:
            process_single_session_intervals(session.id)
            
    except Exception as e:
        logger.error(f"Error querying active sessions: {e}")

def process_single_session_intervals(session_id: str):
    db = get_db()
    
    # Fetch unprocessed readings
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
            t = float(data.get("temperature", 0.0))
            h = float(data.get("humidity", 0.0))
            l = float(data.get("light", 0.0))
            n = float(data.get("sound_level", 0.0))
            
            # Physics Score
            base_score = compute_rule_score(t, h, n, l)
            rule_scores.append(base_score)

            features_batch.append([t, h, n, l])
            doc_ids.append(doc.id)
        except Exception:
            continue

    if not features_batch: return

    # ML Residual Prediction
    residuals = predict_batch(features_batch)
    
    batch = db.batch()
    for i, doc_id in enumerate(doc_ids):
        final_score = max(0.0, min(100.0, rule_scores[i] + residuals[i]))
        
        # Write Interval Report
        score_ref = db.collection(COLLECTION_SCORES).document() 
        batch.set(score_ref, {
            "session_id": session_id,
            "reading_id": doc_id,
            "score": float(final_score),
            "created_at": datetime.now(timezone.utc)
        })
        
        # Mark Processed
        reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
        batch.update(reading_ref, {KEY_PROCESSED: True})

    batch.commit()
    logger.info(f"✅ Processed {len(docs)} intervals for {session_id}")

# ---------------------------------------------------------
# 2. END-OF-SESSION AGGREGATOR (NEW FEATURE)
# ---------------------------------------------------------
def process_finished_sessions():
    """
    Scans for sessions marked "END" that do NOT have a summary yet.
    Calculates averages and writes to 'session_records'.
    """
    db = get_db()
    if db is None: return

    try:
        # Find ended sessions
        ended_sessions = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ENDED))\
            .stream()

        for session in ended_sessions:
            try:
                # IDEMPOTENCY CHECK: Do not re-summarize if record exists
                summary_ref = db.collection(COLLECTION_SUMMARY).document(session.id)
                if summary_ref.get().exists:
                    continue 

                logger.info(f"∑ Generating Summary for ended session: {session.id}")
                generate_session_summary(session.id, summary_ref)
                
            except Exception as e:
                logger.error(f"Failed to summarize session {session.id}: {e}")

    except Exception as e:
        logger.error(f"Error scanning finished sessions: {e}")

def generate_session_summary(session_id: str, target_ref):
    db = get_db()
    
    # 1. Fetch ALL Readings (for averages & duration)
    readings = list(db.collection(COLLECTION_READINGS)\
        .where(filter=FieldFilter("session_id", "==", session_id))\
        .stream())
    
    if not readings:
        logger.warning(f"Session {session_id} has no readings. Skipping.")
        return

    # 2. Extract Data
    temps, hums, lights, sounds, timestamps = [], [], [], [], []
    
    for r in readings:
        d = r.to_dict()
        temps.append(d.get("temperature", 0))
        hums.append(d.get("humidity", 0))
        lights.append(d.get("light", 0))
        sounds.append(d.get("sound_level", 0))
        
        # Handle Timestamp (Supports Firestore Timestamp or ISO String)
        ts = d.get("timestamp")
        if ts:
            # If it's a string, try to parse; if it's already datetime object, use it
            if isinstance(ts, str):
                try: timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except: pass
            else:
                timestamps.append(ts)

    # 3. Fetch Scores (for Sleep Quality)
    scores_docs = list(db.collection(COLLECTION_SCORES)\
        .where(filter=FieldFilter("session_id", "==", session_id))\
        .stream())
    
    scores = [s.to_dict().get("score", 0) for s in scores_docs]
    avg_quality = float(np.mean(scores)) if scores else 0.0

    # 4. Calculate Duration
    duration_seconds = 0
    if timestamps:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        duration_seconds = int((max_ts - min_ts).total_seconds())

    # 5. Prepare The Record (Matching your exact schema)
    summary_data = {
        "averageTemperature": float(np.mean(temps)) if temps else 0.0,
        "averageHumidity": float(np.mean(hums)) if hums else 0.0,
        "averageLightExposure": float(np.mean(lights)) if lights else 0.0,
        "averageSoundLevel": float(np.mean(sounds)) if sounds else 0.0,
        "sleepQualityScore": avg_quality,
        "totalSleepDuration": duration_seconds,
        "date": datetime.now(timezone.utc) # Time the summary was generated
    }
    
    # 6. Write it
    target_ref.set(summary_data)
    logger.info(f"🎉 Summary written for {session_id}: Score {avg_quality:.1f}")