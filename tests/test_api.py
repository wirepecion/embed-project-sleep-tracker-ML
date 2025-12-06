# tests/test_api.py
import os
import json
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

# import the app and model_service from your scaffold
from fastapi_ml_scaffold import app, model_service, MODEL_CONFIG

client = TestClient(app)

# default headers (scaffold expects X-API-Key for normal endpoints)
API_KEY = "test-key"
ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "admin-secret")

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def test_health():
    r = client.get("/v1/health")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j and j["status"] == "ok"
    assert "model_loaded" in j

def test_model_info():
    r = client.get("/v1/model/info")
    assert r.status_code == 200
    j = r.json()
    assert "model_version" in j and "features" in j

def test_score_interval_minimal():
    payload = {
        "device_id": "esp32-test",
        "timestamp": iso_now(),
        "temperature": 25.0,
        "humidity": 65.0,
        "light": 3.0,
        "sound_level": 34.0
    }
    r = client.post("/v1/score/interval", headers={"X-API-Key": API_KEY}, json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "interval_score" in j
    assert "rule_score" in j
    # If no model is loaded, residual should be None
    if model_service.model is None:
        assert j["residual"] is None or j["model_version"] is None

def test_score_interval_missing_api_key():
    payload = {
        "device_id": "esp32-test",
        "timestamp": iso_now(),
        "temperature": 25.0,
        "humidity": 65.0,
        "light": 3.0,
        "sound_level": 34.0
    }
    r = client.post("/v1/score/interval", json=payload)  # no API key
    assert r.status_code == 401 or r.status_code == 422

def test_score_interval_batch():
    recs = []
    for i in range(3):
        recs.append({
            "device_id": f"esp32-{i}",
            "timestamp": iso_now(),
            "temperature": 24.0 + i,
            "humidity": 60.0 + i,
            "light": 2.0 + i,
            "sound_level": 30.0 + i
        })
    r = client.post("/v1/score/interval/batch", headers={"X-API-Key": API_KEY}, json={"records": recs})
    assert r.status_code == 200
    j = r.json()
    assert "results" in j and j["processed"] == len(j["results"])

def test_score_session_minimal():
    payload = {
        "session_id": "s-test-1",
        "device_id": "esp32-test",
        "start_ts": iso_now(),
        "end_ts": iso_now(),
        "total_sleep_duration_minutes": 420,
        "avg_temperature": 25.0,
        "avg_humidity": 65.0,
        "avg_light": 3.0,
        "avg_sound_level": 34.0,
        "num_intervals": 84
    }
    r = client.post("/v1/score/session", headers={"X-API-Key": API_KEY}, json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "session_score" in j and "breakdown" in j

def test_model_reload_protected_wrong_key():
    r = client.post("/v1/model/reload", headers={"admin_key": "wrong"})
    assert r.status_code == 401

def test_model_reload_admin_key():
    # call with correct admin key (default in scaffold is "admin-secret" or env ADMIN_API_KEY)
    r = client.post("/v1/model/reload", headers={"admin_key": ADMIN_KEY})
    assert r.status_code == 200
    j = r.json()
    assert "status" in j

def test_debug_score_requires_admin_key():
    payload = {
        "device_id": "esp32-test",
        "timestamp": iso_now(),
        "temperature": 25.0,
        "humidity": 65.0,
        "light": 3.0,
        "sound_level": 34.0
    }
    r = client.post("/v1/debug/score", headers={"admin_key": "wrong"}, json=payload)
    assert r.status_code == 401

def test_debug_score_with_admin_key():
    payload = {
        "device_id": "esp32-test",
        "timestamp": iso_now(),
        "temperature": 25.0,
        "humidity": 65.0,
        "light": 3.0,
        "sound_level": 34.0
    }
    r = client.post("/v1/debug/score", headers={"admin_key": ADMIN_KEY}, json=payload)
    # debug is dev-only but scaffold will respond if admin_key correct
    assert r.status_code == 200
    j = r.json()
    assert "rule_score" in j and "contributions" in j
