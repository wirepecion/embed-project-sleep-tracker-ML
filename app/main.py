# app/main.py
import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# local helpers (keep these modules in your repo)
from app.firebase_client import init_firebase  # returns Firestore client (or None)
from app.model_loader import predict            # predict(features:list[float]) -> float

# config (tune via env)
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "30"))
SESSION_INACTIVITY_MINUTES = int(os.environ.get("SESSION_INACTIVITY_MINUTES", "60"))
PROCESS_BATCH_LIMIT = int(os.environ.get("PROCESS_BATCH_LIMIT", "200"))
MAX_SESSIONS_PER_PASS = int(os.environ.get("MAX_SESSIONS_PER_PASS", "100"))
API_KEY_REQUIRED = os.environ.get("API_KEY_REQUIRED", "false").lower() == "true"

app = FastAPI(title="Sleep ML Service (poller + push)")

# ---------- simple timestamp parser ----------
def parse_to_utc_iso(ts_str: str) -> str:
    """
    Accept ISO8601 or human-like strings; return ISO8601 UTC with Z.
    """
    # we avoid extra dependencies; try fromisoformat first
    try:
        if isinstance(ts_str, str) and ts_str.endswith("Z"):
            # already ISO UTC
            return ts_str
        # allow python iso parsing
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.replace(tzinfo=None).isoformat() + "Z"
    except Exception:
        # fallback: try dateutil if available
        try:
            from dateutil import parser as _parser
            dt = _parser.parse(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.replace(tzinfo=None).isoformat() + "Z"
        except Exception:
            # last resort: current time
            return datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"

# ---------- Pydantic models ----------
class EnvWritten(BaseModel):
    session_id: str
    timestamp: str
    client_ingest_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

# ---------- Firestore helpers (set at startup) ----------
db = None  # firestore client to be set on startup

def env_doc_ref(session_id: str, iso_ts: str):
    return db.collection("sessions").document(session_id).collection("environment").document(iso_ts)

def ml_doc_ref(session_id: str, iso_ts: str):
    return db.collection("sessions").document(session_id).collection("ml_scores").document(iso_ts)

def session_summary_ref(session_id: str):
    return db.collection("sessions").document(session_id).collection("meta").document("summary")

# ---------- prediction logic ----------
def compute_score_from_env(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    env: { temp_c, humidity_pct, light_lux, noise_db }
    returns dict with interval_score, model_version (if available), confidence, rule_score, residual
    """
    # build features list expected by your model_loader.predict
    # adjust order to match training
    temp = env.get("temp_c")
    hum = env.get("humidity_pct")
    light = env.get("light_lux")
    noise = env.get("noise_db")

    # attempt model prediction (requires app.model_loader.predict)
    model_version = None
    confidence = 0.5
    interval_score = None
    residual = 0.0
    rule_score = None

    # fallback deterministic rule (simple)
    def rule_score_fn(t, h, n, l):
        score = 100.0
        if t is not None:
            score -= 3.5 * abs(t - 20.0)
        if h is not None:
            score -= 0.4 * abs(h - 50.0)
        if n is not None and n > 30:
            score -= 2.0 * (n - 30.0)
        if l is not None and l > 1:
            score -= 1.5 * max(0.0, l - 1.0)
        return float(max(0.0, min(100.0, score)))

    rule_score = rule_score_fn(temp, hum, noise, light)

    # Try model predict: expect list of floats in same order as training
    try:
        features = [
            temp if temp is not None else 20.0,
            hum if hum is not None else 50.0,
            light if light is not None else 0.0,
            noise if noise is not None else 25.0
        ]
        pred_val = predict(features)  # should return float
        # If your model predicts residual, you may want to combine. Here assume model predicts final score.
        interval_score = float(pred_val)
        model_version = os.environ.get("MODEL_VERSION", "model_joblib")
        confidence = 0.8
    except Exception as e:
        # model failed -> fallback to rule
        interval_score = float(rule_score)
        model_version = None
        confidence = 0.45

    return {
        "interval_score": interval_score,
        "rule_score": rule_score,
        "residual": residual,
        "model_version": model_version,
        "confidence": confidence
    }

# ---------- process single env doc ----------
def process_env_doc(session_id: str, timestamp_raw: str) -> Dict[str, Any]:
    """
    Idempotent processor: reads env doc, writes ml_scores doc, marks env processed.
    timestamp_raw accepted as human or ISO string.
    """
    iso_ts = parse_to_utc_iso(timestamp_raw)
    env_ref = env_doc_ref(session_id, iso_ts)
    env_snap = env_ref.get()
    if not env_snap.exists:
        raise HTTPException(status_code=404, detail=f"environment doc not found {session_id} @ {iso_ts}")

    env = env_snap.to_dict()

    # idempotency: if ml doc exists, return it
    ml_ref = ml_doc_ref(session_id, iso_ts)
    ml_snap = ml_ref.get()
    if ml_snap.exists:
        out = ml_snap.to_dict()
        out["status"] = "already_processed"
        return out

    # compute score
    pred = compute_score_from_env(env)

    ml_payload = {
        "timestamp": iso_ts,
        "temp_c": env.get("temp_c"),
        "humidity_pct": env.get("humidity_pct"),
        "light_lux": env.get("light_lux"),
        "noise_db": env.get("noise_db"),
        "interval_score": pred["interval_score"],
        "rule_score": pred.get("rule_score"),
        "residual": pred.get("residual"),
        "model_version": pred.get("model_version"),
        "confidence": pred.get("confidence"),
        "client_ingest_id": env.get("client_ingest_id"),
        "prediction_type": "realtime",
        "created_at": db.SERVER_TIMESTAMP if hasattr(db, "SERVER_TIMESTAMP") else None
    }

    # write ml_scores
    ml_ref.set(ml_payload)

    # mark env processed (attempt update; if missing field, set it)
    try:
        env_ref.update({
            "_ml_processed": True,
            "_ml_processed_at": db.SERVER_TIMESTAMP
        })
    except Exception:
        # some SDK setups require a different SERVER_TIMESTAMP symbol; ignore if update fails
        pass

    result = {"status": "processed", "firebase_path": f"sessions/{session_id}/ml_scores/{iso_ts}", **ml_payload}
    return result

# ---------- session summary ----------
def compute_and_write_session_summary(session_id: str) -> Dict[str, Any]:
    ml_collection = db.collection("sessions").document(session_id).collection("ml_scores")
    docs = list(ml_collection.stream())
    if not docs:
        return {"status": "no_intervals"}

    scores = []
    temps = []
    hums = []
    lights = []
    noises = []
    timestamps = []
    for d in docs:
        dd = d.to_dict()
        if "interval_score" in dd:
            scores.append(dd["interval_score"])
            temps.append(dd.get("temp_c"))
            hums.append(dd.get("humidity_pct"))
            lights.append(dd.get("light_lux"))
            noises.append(dd.get("noise_db"))
            timestamps.append(dd.get("timestamp"))

    import numpy as np
    scores_arr = np.array(scores)
    summary = {
        "session_id": session_id,
        "avg_comfort_score": float(np.mean(scores_arr)),
        "min_score": float(np.min(scores_arr)),
        "max_score": float(np.max(scores_arr)),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "num_intervals": int(len(scores_arr)),
        "stats": {
            "temp_mean": float(np.nanmean([x for x in temps if x is not None])) if any(temps) else None,
            "humidity_mean": float(np.nanmean([x for x in hums if x is not None])) if any(hums) else None,
            "light_mean": float(np.nanmean([x for x in lights if x is not None])) if any(lights) else None,
            "noise_mean": float(np.nanmean([x for x in noises if x is not None])) if any(noises) else None,
        },
        "created_at": db.SERVER_TIMESTAMP
    }

    session_summary_ref(session_id).set(summary)
    return {"status": "summary_written", "path": f"sessions/{session_id}/meta/summary", "summary": summary}

# ---------- detect session inactivity ----------
def session_is_inactive(session_id: str, inactivity_minutes: int = SESSION_INACTIVITY_MINUTES) -> bool:
    env_coll = db.collection("sessions").document(session_id).collection("environment")
    q = env_coll.order_by("timestamp", direction=db.Query.DESCENDING).limit(1).stream()
    last = None
    for d in q:
        last = d.to_dict()
        break
    if not last or "timestamp" not in last:
        return False
    try:
        last_ts = last["timestamp"]
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except Exception:
        # treat as not inactive to avoid false summaries
        return False
    age = datetime.utcnow().replace(tzinfo=timezone.utc) - last_dt
    return age > timedelta(minutes=inactivity_minutes)

# ---------- poller ----------
async def poller_loop():
    if db is None:
        app.logger = getattr(app, "logger", None)
        print("Poller disabled: Firestore not configured")
        return

    print("Poller started; interval:", POLL_INTERVAL_SECONDS)
    while True:
        try:
            # list sessions (top-level docs)
            sessions = db.collection("sessions").list_documents()
            processed = 0
            for sref in sessions:
                session_id = sref.id
                if processed >= MAX_SESSIONS_PER_PASS:
                    break

                # find env docs not processed
                env_coll = sref.collection("environment")
                # try direct query for _ml_processed == False
                unprocessed = []
                try:
                    q = env_coll.where("_ml_processed", "==", False).order_by("timestamp").limit(PROCESS_BATCH_LIMIT).stream()
                    unprocessed = list(q)
                except Exception:
                    # Firestore may not allow where on missing fields — fallback to scanning
                    docs = env_coll.order_by("timestamp").limit(PROCESS_BATCH_LIMIT).stream()
                    for d in docs:
                        dd = d.to_dict()
                        if not dd.get("_ml_processed", False):
                            unprocessed.append(d)

                for d in unprocessed:
                    iso_ts = d.id
                    try:
                        res = process_env_doc(session_id, iso_ts)
                        # optional: log res
                    except Exception as e:
                        print("Error processing", session_id, iso_ts, e)
                        continue

                # check inactivity to compute session summary
                try:
                    if session_is_inactive(session_id):
                        meta_ref = session_summary_ref(session_id).get()
                        if not meta_ref.exists:
                            compute_and_write_session_summary(session_id)
                except Exception as e:
                    print("Error computing summary for", session_id, e)

                processed += 1

            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            print("Poller loop exception:", e)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

# ---------- endpoints ----------
def check_api_key(x_api_key: Optional[str]):
    if API_KEY_REQUIRED and not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key required")

@app.post("/v1/process/env_doc")
async def process_env_doc_endpoint(payload: EnvWritten, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    # safe call to process a single env doc immediately (push-trigger)
    global db
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore not configured")
    try:
        res = process_env_doc(payload.session_id, payload.timestamp)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-firestore")
async def test_firestore():
    global db
    if db is None:
        return {"status": "error", "message": "Firestore not initialized"}
    try:
        ref = db.collection("test_collection").document("connectivity_test")
        ref.set({"message": "hello", "ts": datetime.utcnow().isoformat()})
        snap = ref.get()
        return {"status": "success", "doc": snap.to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/v1/score/session-summary")
async def session_summary_endpoint(req: SessionRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore not configured")
    try:
        res = compute_and_write_session_summary(req.session_id)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- startup ----------
@app.on_event("startup")
async def startup_event():
    global db
    db = init_firebase()
    if db is None:
        print("Firestore not configured. Poller disabled.")
        return
    # warm model by trying a test predict if needed (model_loader handles its own loading)
    print("Firestore initialized. Starting poller...")
    # start poller in background
    asyncio.create_task(poller_loop())
