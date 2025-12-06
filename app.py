# app.py
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from utils_time import parse_to_utc_iso, iso_to_epoch_ms
from firebase_init import init_firebase
from model_service import hybrid_predict, rule_score, res_model
from firebase_writes import write_environment_record, write_ml_score
from dateutil import parser
import numpy as np
import statistics

# init firebase
db = init_firebase()

app = FastAPI(title="Sleep ML (5-min) Service")

# Pydantic schemas (no device_id; single device)
class IntervalPayload(BaseModel):
    session_id: str
    timestamp: str          # accept human string or ISO
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    light_lux: Optional[float] = None
    noise_db: Optional[float] = None
    client_ingest_id: Optional[str] = None

class BatchPayload(BaseModel):
    session_id: str
    records: List[IntervalPayload]

class SessionSummaryRequest(BaseModel):
    session_id: str

# helpers to normalize timestamp & payload
def normalize_and_validate(p: IntervalPayload):
    # parse timestamp, convert to ISO UTC
    try:
        iso = parse_to_utc_iso(p.timestamp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad timestamp: {e}")
    # build environment payload
    env = {
        "timestamp": iso,
        "temp_c": p.temp_c,
        "humidity_pct": p.humidity_pct,
        "light_lux": p.light_lux,
        "noise_db": p.noise_db,
        "client_ingest_id": p.client_ingest_id
    }
    return iso, env

# endpoint: single interval
@app.post("/v1/score/interval")
def score_interval(payload: IntervalPayload, x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    iso_ts, env = normalize_and_validate(payload)

    # write environment raw (optional; your backend may already do it)
    write_environment_record(db, payload.session_id, iso_ts, env)

    # predict
    out = hybrid_predict(env["temp_c"], env["humidity_pct"], env["noise_db"], env["light_lux"])

    ml_payload = {
        "timestamp": iso_ts,
        "comfort_score": out["interval_score"],
        "rule_score": out["rule_score"],
        "residual": out["residual"],
        "model_version": out["model_version"],
        "confidence": out["confidence"],
        "input": env
    }

    written, path = write_ml_score(db, payload.session_id, iso_ts, ml_payload)

    return { "timestamp": iso_ts, "written": written, "firebase_path": path, **out }

# endpoint: batch intervals
@app.post("/v1/score/interval/batch")
def score_batch(payload: BatchPayload, x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    results = []
    for rec in payload.records:
        iso_ts, env = normalize_and_validate(rec)
        write_environment_record(db, payload.session_id, iso_ts, env)
        out = hybrid_predict(env["temp_c"], env["humidity_pct"], env["noise_db"], env["light_lux"])
        ml_payload = {
            "timestamp": iso_ts,
            "comfort_score": out["interval_score"],
            "rule_score": out["rule_score"],
            "residual": out["residual"],
            "model_version": out["model_version"],
            "confidence": out["confidence"],
            "input": env
        }
        written, path = write_ml_score(db, payload.session_id, iso_ts, ml_payload)
        results.append({ "timestamp": iso_ts, "written": written, "firebase_path": path, **out })
    return {"processed": len(results), "results": results}

# endpoint: session summary
@app.post("/v1/score/session-summary")
def session_summary(req: SessionSummaryRequest, x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    session_id = req.session_id
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore client not available")

    coll = db.collection("sessions").document(session_id).collection("ml_scores")
    docs = coll.stream()
    scores = []
    temps = []
    hums = []
    lights = []
    noises = []
    timestamps = []
    for d in docs:
        data = d.to_dict()
        if "comfort_score" in data:
            scores.append(data["comfort_score"])
            inp = data.get("input", {})
            temps.append(inp.get("temp_c"))
            hums.append(inp.get("humidity_pct"))
            lights.append(inp.get("light_lux"))
            noises.append(inp.get("noise_db"))
            timestamps.append(data.get("timestamp"))

    if not scores:
        raise HTTPException(status_code=404, detail="No ml_scores found for session")

    summary = {
        "session_id": session_id,
        "avg_comfort_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "model_version": res_model.version,
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
        }
    }

    # write summary doc
    doc_ref = db.collection("sessions").document(session_id).collection("meta").document("summary")
    doc_ref.set({ **summary, "created_at": firestore.SERVER_TIMESTAMP })

    return summary

# model info & health
@app.get("/v1/model/info")
def model_info():
    return {"model_version": res_model.version, "loaded": res_model.model is not None, "features": ["temp_c","humidity_pct","light_lux","noise_db"]}

@app.get("/v1/health")
def health():
    return {"status": "ok", "model_loaded": res_model.model is not None}
