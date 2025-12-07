from datetime import datetime, timezone, timedelta
import logging
import numpy as np
import requests  # <--- NEW IMPORT
from google.cloud.firestore import FieldFilter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os  

import app.firebase_client as fb_client
from app.model_loader import predict_batch
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

# --- DIFFUSER CONFIG ---
DIFFUSER_THRESHOLD_SCORE = 90.0  
BLYNK_AUTH_TOKEN = "y9gtpw7iauYC0CJSNe2JHwOjznVsrBTi"
BLYNK_URL = "https://blynk.cloud/external/api/update?token={token}&V0={value}"

def get_db():
    return fb_client.db

# --- HELPER: CONTROL DIFFUSER ---
def set_diffuser_state(is_on: bool):
    """
    Calls Blynk API to Turn Diffuser ON (1) or OFF (0)
    """
    val = 1 if is_on else 0
    url = BLYNK_URL.format(token=BLYNK_AUTH_TOKEN, value=val)
    try:
        # Timeout is important so your server doesn't freeze if Blynk is down
        resp = requests.get(url, timeout=5) 
        if resp.status_code == 200:
            logger.info(f"💨 Diffuser set to {val} (ON)" if is_on else f"🛑 Diffuser set to {val} (OFF)")
        else:
            logger.warning(f"Blynk API Error: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failed to call Blynk: {e}")

# ---------------------------------------------------------
# 1. REAL-TIME INTERVAL PROCESSING
# ---------------------------------------------------------
# In services.py

def process_active_sessions():
    db = get_db()
    if db is None: db = fb_client.init_firebase()

    # 1. Query for unprocessed readings directly
    # We don't care which session they belong to yet
    unprocessed_docs = db.collection(COLLECTION_READINGS)\
        .where(filter=FieldFilter("is_processed", "==", False))\
        .limit(50)\
        .stream()

    # Group them by session_id in memory to process batches efficiently
    sessions_map = {}
    doc_count = 0
    
    for doc in unprocessed_docs:
        data = doc.to_dict()
        sid = data.get("session_id")
        if sid:
            if sid not in sessions_map: sessions_map[sid] = []
            sessions_map[sid].append(doc)
            doc_count += 1

    if doc_count == 0:
        logger.info("No new readings to process.")
        return # Cost: Only 1 read!

    logger.info(f"Found {doc_count} readings across {len(sessions_map)} sessions.")

    # 2. Process each session found
    for session_id, docs in sessions_map.items():
        # You can reuse your existing logic here, just pass the docs directly
        # to avoid re-querying!
        process_specific_readings(session_id, docs)

# --- EMAIL CONFIG ---
GMAIL_SENDER = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
GMAIL_RECEIVER = os.getenv("GMAIL_RECEIVER")

def send_summary_email(summary_data):
    """
    Sends a formatted HTML email with the sleep session results.
    """
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        logger.warning("⚠️ Email not configured. Skipping notification.")
        return

    try:
        # 1. Setup Message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_SENDER
        msg['To'] = GMAIL_RECEIVER
        
        # Subject: e.g. "Sleep Report: 92% Quality (8h 15m)"
        score = summary_data.get("sleepQualityScore", 0)
        duration_s = summary_data.get("totalSleepDuration", 0)
        hours = duration_s // 3600
        mins = (duration_s % 3600) // 60
        
        msg['Subject'] = f"🌙 Sleep Report: {score}% Quality ({hours}h {mins}m)"

        # 2. HTML Body
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
              <h2 style="color: #4CAF50; text-align: center;">Sleep Session Summary</h2>
              
              <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center;">
                <h1 style="font-size: 48px; margin: 0; color: #333;">{score}</h1>
                <p style="margin: 0; color: #666;">Sleep Quality Score</p>
              </div>

              <table style="width: 100%; margin-top: 20px; border-collapse: collapse;">
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Duration:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{hours}h {mins}m</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Avg Temp:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{summary_data.get('averageTemperature', 0):.1f}°C</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Avg Humidity:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{summary_data.get('averageHumidity', 0):.1f}%</td>
                </tr>
                <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Avg Noise:</strong></td>
                  <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">{summary_data.get('averageSoundLevel', 0):.1f} dB</td>
                </tr>
              </table>
              
              <p style="text-align: center; margin-top: 30px; font-size: 12px; color: #999;">
                Session ID: {summary_data.get('session_id')}
              </p>
            </div>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))

        # 3. Connect to Gmail SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure connection
        server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(GMAIL_SENDER, GMAIL_RECEIVER, text)
        server.quit()
        
        logger.info(f"📧 Email sent to {GMAIL_RECEIVER}")

    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")

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

    # 4. Get AI Prediction
    residuals = predict_batch(features_batch)
    
    batch = db.batch()
    
    # FLAG: Track if we need to turn on the diffuser
    should_open_diffuser = False

    for i, doc_id in enumerate(doc_ids):
        # HYBRID LOGIC: Final = Base + Residual
        final_score = rule_scores[i] + residuals[i]
        final_score = max(0.0, min(100.0, final_score))
        
        # --- NEW LOGIC: CHECK THRESHOLD ---
        if final_score < DIFFUSER_THRESHOLD_SCORE:
            should_open_diffuser = True
        # ----------------------------------

        # Write Interval Report
        score_ref = db.collection(COLLECTION_SCORES).document() 
        batch.set(score_ref, {
            "session_id": session_id,
            "sleep_score": float(final_score),
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Mark Processed
        reading_ref = db.collection(COLLECTION_READINGS).document(doc_id)
        batch.update(reading_ref, {KEY_PROCESSED: True})

    batch.commit()
    logger.info(f"✅ Processed {len(docs)} intervals for {session_id}")

    # --- ACTION: TRIGGER HARDWARE ---
    # We trigger ONCE per batch. 
    # If ANY reading in the last 5-50 mins was bad, we blast the scent.
    if should_open_diffuser:
        logger.info(f"📉 Low Score Detected (<{DIFFUSER_THRESHOLD_SCORE}%). Activating Diffuser...")
        set_diffuser_state(is_on=True)
    else:
        # OPTIONAL: Turn it off if sleep is good? 
        # Uncomment the next line if you want it to auto-close when sleep improves.
        # set_diffuser_state(is_on=False) 
        pass

# ---------------------------------------------------------
# 2. END-OF-SESSION AGGREGATOR
# ---------------------------------------------------------
def process_finished_sessions():
    db = get_db()
    if db is None: return

    lookback_window = datetime.now(timezone.utc) - timedelta(minutes=20)

    try:
        ended_sessions = db.collection(COLLECTION_SESSIONS)\
            .where(filter=FieldFilter(KEY_SESSION_STATUS, "==", VAL_SESSION_ENDED))\
            .where(filter=FieldFilter("timestamp", ">", lookback_window))\
            .stream()

        for session in ended_sessions:
            try:
                summary_ref = db.collection(COLLECTION_SUMMARY).document(session.id)
                if summary_ref.get().exists:
                    continue 

                logger.info(f"∑ Processing Finished Session: {session.id}")
                
                # Flush pending readings (This will also trigger diffuser if last reading was bad!)
                process_single_session_intervals(session.id)

                generate_session_summary(session.id, summary_ref)
                
            except Exception as e:
                logger.error(f"Failed to summarize session {session.id}: {e}")

    except Exception as e:
        logger.error(f"Error scanning finished sessions: {e}")

def generate_session_summary(session_id: str, target_ref):
    db = get_db()
    
    session_doc = db.collection(COLLECTION_SESSIONS).document(session_id).get()
    if not session_doc.exists: return

    session_data = session_doc.to_dict()
    
    duration_seconds = 0
    try:
        def to_dt(val):
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return val 

        start_time = to_dt(session_data.get("timestamp"))
        end_time = to_dt(session_data.get("ended_at"))

        if start_time and end_time:
            duration_seconds = int((end_time - start_time).total_seconds())
    except Exception as e:
        logger.warning(f"Could not calc duration: {e}")

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

    scores_docs = list(db.collection(COLLECTION_SCORES)\
        .where(filter=FieldFilter("session_id", "==", session_id))\
        .stream())
    
    scores = [s.to_dict().get("score", 0) for s in scores_docs] # Note: field name in COLLECTION_SCORES is 'sleep_score' in your new code?
    # FIX: In process_single_session_intervals you wrote 'sleep_score', check key!
    # Let's be safe and check both or fix the writer above.
    # The writer above uses 'sleep_score'.
    scores = [s.to_dict().get("sleep_score", 0) for s in scores_docs]
    
    avg_quality = float(np.mean(scores)) if scores else 0.0

    summary_data = {
        "session_id": session_id,
        "averageTemperature": float(np.mean(temps)) if temps else 0.0,
        "averageHumidity": float(np.mean(hums)) if hums else 0.0,
        "averageLightExposure": float(np.mean(lights)) if lights else 0.0,
        "averageSoundLevel": float(np.mean(sounds)) if sounds else 0.0,
        "sleepQualityScore": float(f"{avg_quality:.1f}"),
        "totalSleepDuration": duration_seconds, 
        "date": datetime.now(timezone.utc)
    }
    
    target_ref.set(summary_data)
    logger.info(f"🎉 Summary written for {session_id}: Duration {duration_seconds}s")

