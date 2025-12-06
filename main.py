# main.py
import os
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from utils_time import parse_to_utc_iso
from firebase_client import init_firebase
from model_service import hybrid_predict, ResidualModel
from firebase_admin import firestore

# config
ENABLE_POLLER = os.environ.get("ENABLE_FIRESTORE_POLLER", "true").lower() == "true"
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "30"))
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin-secret")
API_KEY_REQUIRED = os.environ.get("API_KEY_REQUIRED", "false").lower() == "true"

app = FastAPI(title="Sleep Tracker ML Service (Firestore-driven)")



# Pydantic models
class EnvWritten(BaseModel):
    session_id: str
    timestamp: str
    client_ingest_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

# init placeholders
db = None

# auth helper
def check_api_key(x_api_key: Optional[str]):
    if API_KEY_REQUIRED and not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key required")

# helper doc refs
def env_doc_ref(session_id: str, iso_ts: str):
    return db.collection("sessions").document(session_id).collection("environment").document(iso_ts)

def ml_doc_ref(session_id: str, iso_ts: str):
    return db.collection("sessions").document(session_id).collection("ml_scores").document(iso_ts)

# core processing: read env doc -> predict -> write ml doc
def process_env_doc(session_id: str, timestamp_raw: str) -> Dict[str, Any]:
    iso_ts = parse_to_utc_iso(timestamp_raw)
    env_ref = env_doc_ref(session_id, iso_ts)
    env_snap = env_ref.get()
    if not env_snap.exists:
        raise HTTPException(status_code=404, detail=f"environment doc not found for {session_id}@{iso_ts}")
    env = env_snap.to_dict()

    # fields (impute None allowed)
    temp = env.get("temp_c")
    hum = env.get("humidity_pct")
    light = env.get("light_lux")
    noise = env.get("noise_db")
    client_ingest_id = env.get("client_ingest_id")

    # idempotency: if ml doc exists, return it
    ml_ref = ml_doc_ref(session_id, iso_ts)
    if ml_ref.get().exists:
        snap = ml_ref.get()
        return {"status": "already_processed", **snap.to_dict()}

    out = hybrid_predict(temp, hum, noise, light)

    ml_payload = {
        "timestamp": iso_ts,
        "temp_c": temp,
        "humidity_pct": hum,
        "light_lux": light,
        "noise_db": noise,
        "interval_score": out["interval_score"],
        "rule_score": out["rule_score"],
        "residual": out["residual"],
        "model_version": out["model_version"],
        "confidence": out["confidence"],
        "client_ingest_id": client_ingest_id,
        "prediction_type": "realtime",
        "created_at": firestore.SERVER_TIMESTAMP
    }

    ml_ref.set(ml_payload)
    return {"status": "processed", "firebase_path": f"sessions/{session_id}/ml_scores/{iso_ts}", **ml_payload}

# endpoint called by backend after env doc is written
@app.post("/v1/process/env_doc")
async def process_env_doc_endpoint(payload: EnvWritten, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    global db
    db = init_firebase()
    try:
        res = process_env_doc(payload.session_id, payload.timestamp)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# test endpoint
@app.get("/test-firestore")
async def test_firestore():
    global db
    db = init_firebase()
    if db is None:
        return {"status": "error", "message": "Firestore not initialized"}
    ref = db.collection("test_collection").document("connectivity_test")
    ref.set({"message": "hello", "created_at": firestore.SERVER_TIMESTAMP})
    doc = ref.get()
    if doc.exists:
        return {"status": "success", "doc": doc.to_dict()}
    return {"status": "error", "message": "write failed"}

# session summary endpoint (reads ml_scores and writes summary)
@app.post("/v1/score/session-summary")
async def session_summary(req: SessionRequest, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    global db
    db = init_firebase()
    session_id = req.session_id
    ml_coll = db.collection("sessions").document(session_id).collection("ml_scores")
    docs = list(ml_coll.stream())
    if not docs:
        raise HTTPException(status_code=404, detail="No ml_scores for session")
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
    summary = {
        "session_id": session_id,
        "avg_comfort_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "model_version": None,  # could set from ResidualModel
        "comfort_trend": "increasing" if scores[-1] > scores[0] else ("decreasing" if scores[-1] < scores[0] else "stable"),
        "stats": {
            "temp_mean": float(np.nanmean([x for x in temps if x is not None])) if any(temps) else None,
            "temp_std": float(np.nanstd([x for x in temps if x is not None])) if any(temps) else None,
            "humidity_mean": float(np.nanmean([x for x in hums if x is not None])) if any(hums) else None,
            "humidity_std": float(np.nanstd([x for x in hums if x is not None])) if any(hums) else None,
            "light_mean": float(np.nanmean([x for x in lights if x is not None])) if any(lights) else None,
            "light_std": float(np.nanstd([x for x in lights if x is not None])) if any(lights) else None,
            "noise_mean": float(np.nanmean([x for x in noises if x is not None])) if any(noises) else None,
            "noise_std": float(np.nanstd([x for x in noises if x is not None])) if any(noises) else None
        },
        "num_intervals": len(scores)
    }
    # write summary
    doc_ref = db.collection("sessions").document(session_id).collection("meta").document("summary")
    doc_ref.set({**summary, "created_at": firestore.SERVER_TIMESTAMP})
    return {"written": True, "firebase_path": f"sessions/{session_id}/meta/summary", **summary}

# background poller that finds env docs without ml_scores and processes them
async def poller_task():
    global db
    if db is None:
        print("Poller disabled: Firestore not initialized.")
        return
    print("Poller started.")
    while True:
        try:
            sessions = db.collection("sessions").list_documents()
            for sref in sessions:
                sid = sref.id
                env_coll = sref.collection("environment")
                docs = env_coll.limit(500).stream()
                for d in docs:
                    iso_ts = d.id
                    # check ml exists
                    mlref = sref.collection("ml_scores").document(iso_ts)
                    if mlref.get().exists:
                        continue
                    try:
                        process_env_doc(sid, iso_ts)
                        env_coll.document(iso_ts).update({"_ml_processed_at": firestore.SERVER_TIMESTAMP})
                    except Exception as e:
                        print("Poller error for", sid, iso_ts, e)
            await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            print("Poller inner loop error:", e)
            await asyncio.sleep(POLL_INTERVAL)

@app.on_event("startup")
async def startup_event():
    global db
    db = init_firebase()
    # start poller if enabled
    if ENABLE_POLLER and db is not None:
        asyncio.create_task(poller_task())
    else:
        print("Poller disabled or Firestore not configured.")
