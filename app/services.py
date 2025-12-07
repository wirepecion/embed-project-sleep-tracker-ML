from datetime import datetime, timezone, timedelta
import logging
import numpy as np
from google.cloud.firestore import FieldFilter

import app.firebase_client as fb_client
from app.model_loader import predict_batch
# IMPORT YOUR PHYSICS RULES
from app.sleep_rules import compute_rule_score

logger = logging.getLogger("SleepService")

# --- CONFIGURATION ---
COLLECTION_SESSIONS = "sleep_sessions"
COLLECTION_READINGS = "sensor_readings"
COLLECTION_SCORES = "interval_records"
COLLECTION_SUMMARY = "session_records"

KEY_SESSION_STATUS = "type"      
VAL_SESSION_ACTIVE = "START"
VAL_SESSION_ENDED = "END"     
KEY_PROCESSED = "is_processed"



def get_db():
    return fb_client.db

# ---------------------------------------------------------
# 1. REAL-TIME INTERVAL PROCESSING
# ---------------------------------------------------------
def process_active_sessions():
    db = get_db()
    if db is None: db = fb_client.init_firebase()
    if db is None: return

    # OPTIMIZATION: Ignore "Zombie" sessions older than 24 hours
    # This prevents the query from growing infinitely over years.
    yesterday = datetime.now(timezone.utc) - timedelta(hours=24)

    try:
        active_sessions_ref = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ACTIVE))\
            .where(filter=FieldFilter("timestamp", ">", yesterday))\
            .stream()

        count = 0
        for session in active_sessions_ref:
            process_single_session_intervals(session.id)
            count += 1
        
        if count > 0:
            logger.info(f"Checked {count} active sessions.")
            
    except Exception as e:
        # Note: If you see "Index required", click the link in the logs!
        logger.error(f"Error querying active sessions: {e}")

def process_single_session_intervals(session_id: str):
    db = get_db()
    
    # Fetch unprocessed readings for this session
    # Note: Requires Composite Index (session_id ASC + is_processed ASC)
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
            base_score = compute_rule_score(t, h, n, l)
            rule_scores.append(base_score)

            # 3. Prepare Features for AI
            features_batch.append([t, h, n, l])
            doc_ids.append(doc.id)
        except Exception:
            continue

    if not features_batch: return

    # 4. Get AI Prediction (The Residual)
    residuals = predict_batch(features_batch)
    
    # 5. Save Results
    batch = db.batch()
    
    for i, doc_id in enumerate(doc_ids):
        # HYBRID LOGIC: Final = Base + Residual
        final_score = rule_scores[i] + residuals[i]
        final_score = max(0.0, min(100.0, final_score))
        
        # Write Interval Report
        score_ref = db.collection(COLLECTION_SCORES).document() 
        batch.set(score_ref, {
            "session_id": session_id,
            # "reading_id": doc_id,
            "sleep_score": float(final_score),
            # "base_rule": float(rule_scores[i]),
            # "ai_residual": float(residuals[i]),
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Mark Processed
        reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
        batch.update(reading_ref, {KEY_PROCESSED: True})

    batch.commit()
    logger.info(f"✅ Processed {len(docs)} intervals for {session_id}")

# ---------------------------------------------------------
# 2. END-OF-SESSION AGGREGATOR
# ---------------------------------------------------------
def process_finished_sessions():
    """
    Scans for sessions marked "END" that do NOT have a summary yet.
    """
    db = get_db()
    if db is None: return

    # Lookback window (20 mins is fine)
    lookback_window = datetime.now(timezone.utc) - timedelta(minutes=20)

    try:
        ended_sessions = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ENDED))\
            .where(filter=FieldFilter("timestamp", ">", lookback_window))\
            .stream()

        for session in ended_sessions:
            try:
                # IDEMPOTENCY: Skip if summary already exists
                summary_ref = db.collection(COLLECTION_SUMMARY).document(session.id)
                if summary_ref.get().exists:
                    continue 

                logger.info(f"∑ Processing Finished Session: {session.id}")
                
                # --- CRITICAL FIX: FLUSH PENDING READINGS ---
                # Before summarizing, run the AI one last time to catch any 
                # readings that arrived right before the session ended.
                process_single_session_intervals(session.id)
                # --------------------------------------------

                # Now generate the summary (which will now include the scores we just calculated)
                generate_session_summary(session.id, summary_ref)
                
            except Exception as e:
                logger.error(f"Failed to summarize session {session.id}: {e}")

    except Exception as e:
        logger.error(f"Error scanning finished sessions: {e}")
def generate_session_summary(session_id: str, target_ref):
    db = get_db()
    
    # 1. Fetch Session Document (NEW STEP)
    # We need this to get the official 'timestamp' (Start) and 'ended_at' (End)
    session_doc = db.collection(COLLECTION_SESSIONS).document(session_id).get()
    if not session_doc.exists:
        logger.warning(f"Session {session_id} doc not found during summary.")
        return

    session_data = session_doc.to_dict()
    
    # 2. Calculate Duration from Session Data (The "Correct" Way)
    duration_seconds = 0
    try:
        # Helper to parse Firestore Timestamp or String
        def to_dt(val):
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return val # Assume it's already a datetime/Timestamp

        start_time = to_dt(session_data.get("timestamp"))
        end_time = to_dt(session_data.get("ended_at"))

        if start_time and end_time:
            duration_seconds = int((end_time - start_time).total_seconds())
    except Exception as e:
        logger.warning(f"Could not calc duration from session doc: {e}")

    # 3. Fetch Readings (For Averages)
    readings = list(db.collection(COLLECTION_READINGS)\
        .where(filter=FieldFilter("session_id", "==", session_id))\
        .stream())
    
    if not readings: return

    temps, hums, lights, sounds = [], [], [], []
    
    for r in readings:
        d = r.to_dict()
        temps.append(float(d.get("temperature", 0)))
        hums.append(float(d.get("humidity", 0)))
        lights.append(float(d.get("light", 0)))
        sounds.append(float(d.get("sound_level", 0)))
        
        # Fallback: If session doc failed to give duration, use sensor timestamps
        if duration_seconds == 0 and d.get("timestamp"):
            # (Logic to track min/max sensor time would go here if needed)
            pass

    # 4. Fetch Scores
    scores_docs = list(db.collection(COLLECTION_SCORES)\
        .where(filter=FieldFilter("session_id", "==", session_id))\
        .stream())
    
    scores = [s.to_dict().get("score", 0) for s in scores_docs]
    avg_quality = float(np.mean(scores)) if scores else 0.0

    # 5. Write Summary
    summary_data = {
        "averageTemperature": float(np.mean(temps)) if temps else 0.0,
        "averageHumidity": float(np.mean(hums)) if hums else 0.0,
        "averageLightExposure": float(np.mean(lights)) if lights else 0.0,
        "averageSoundLevel": float(np.mean(sounds)) if sounds else 0.0,
        "sleepQualityScore": float(f"{avg_quality:.1f}"),
        "totalSleepDuration": duration_seconds,  # <--- Using the accurate value
        "date": datetime.now(timezone.utc)
    }
    
    target_ref.set(summary_data)
    logger.info(f"🎉 Summary written for {session_id}: Duration {duration_seconds}s")