"""
Sleep Quality Prediction API
A FastAPI backend that predicts sleep quality based on environmental factors.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import os

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Quality Prediction API",
    description="Predicts sleep quality based on environmental factors",
    version="1.0.0"
)

# Initialize Firebase Admin SDK
try:
    # Check if Firebase is already initialized
    firebase_admin.get_app()
except ValueError:
    # Initialize Firebase only if not already initialized
    # In production, use a service account key file
    # For now, we'll handle the case where credentials might not be available
    cred_path = os.environ.get('FIREBASE_CREDENTIALS_PATH', 'serviceAccountKey.json')
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    else:
        # If no credentials file, Firebase logging will be skipped
        db = None
        print("Warning: Firebase credentials not found. Logging will be skipped.")


class PredictionInput(BaseModel):
    """Input model for sleep quality prediction"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60)
    humidity: float = Field(..., description="Humidity percentage", ge=0, le=100)
    light: float = Field(..., description="Light level in lux", ge=0)
    sound: float = Field(..., description="Sound level in decibels", ge=0, le=200)


class PredictionOutput(BaseModel):
    """Output model for sleep quality prediction"""
    sleep_quality_percent: float = Field(..., description="Predicted sleep quality percentage")
    reasoning: str = Field(..., description="Explanation of the prediction")


def calculate_sleep_quality(temperature: float, humidity: float, light: float, sound: float) -> tuple[float, str]:
    """
    Rule-based model to predict sleep quality.
    This is a placeholder for future ML model.
    
    Optimal conditions for sleep:
    - Temperature: 15-22°C (60-72°F)
    - Humidity: 30-50%
    - Light: 0-10 lux (dark room)
    - Sound: 0-40 dB (quiet)
    
    Returns:
        tuple: (sleep_quality_percent, reasoning)
    """
    score = 100.0
    reasons = []
    
    # Temperature scoring
    if 15 <= temperature <= 22:
        temp_score = 0
        reasons.append("Temperature is optimal")
    elif 12 <= temperature < 15 or 22 < temperature <= 25:
        temp_score = 15
        reasons.append("Temperature is slightly suboptimal")
    else:
        temp_score = 30
        if temperature < 12:
            reasons.append("Temperature is too cold")
        else:
            reasons.append("Temperature is too warm")
    
    # Humidity scoring
    if 30 <= humidity <= 50:
        humidity_score = 0
        reasons.append("Humidity is optimal")
    elif 20 <= humidity < 30 or 50 < humidity <= 60:
        humidity_score = 15
        reasons.append("Humidity is slightly suboptimal")
    else:
        humidity_score = 25
        if humidity < 20:
            reasons.append("Humidity is too low")
        else:
            reasons.append("Humidity is too high")
    
    # Light scoring
    if light <= 10:
        light_score = 0
        reasons.append("Light level is optimal (dark)")
    elif 10 < light <= 50:
        light_score = 15
        reasons.append("Light level is slightly elevated")
    elif 50 < light <= 200:
        light_score = 25
        reasons.append("Light level is too high")
    else:
        light_score = 35
        reasons.append("Light level is way too high")
    
    # Sound scoring
    if sound <= 40:
        sound_score = 0
        reasons.append("Sound level is optimal (quiet)")
    elif 40 < sound <= 60:
        sound_score = 15
        reasons.append("Sound level is slightly elevated")
    elif 60 < sound <= 80:
        sound_score = 25
        reasons.append("Sound level is too high")
    else:
        sound_score = 35
        reasons.append("Sound level is way too high")
    
    # Calculate final score
    score = max(0, score - temp_score - humidity_score - light_score - sound_score)
    
    # Create reasoning string
    reasoning = ". ".join(reasons) + "."
    
    return round(score, 2), reasoning


def log_to_firestore(input_data: dict, output_data: dict):
    """
    Log prediction input and output to Firebase Firestore
    
    Args:
        input_data: Dictionary containing input parameters
        output_data: Dictionary containing prediction results
    """
    if db is None:
        print("Firebase not initialized. Skipping logging.")
        return
    
    try:
        # Create a document in the 'predictions' collection
        doc_ref = db.collection('predictions').document()
        doc_ref.set({
            'input': input_data,
            'output': output_data,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print(f"Logged prediction to Firestore with ID: {doc_ref.id}")
    except Exception as e:
        print(f"Error logging to Firestore: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sleep Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict sleep quality from environmental factors",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_sleep_quality(input_data: PredictionInput):
    """
    Predict sleep quality based on environmental factors.
    
    Args:
        input_data: Environmental factors (temperature, humidity, light, sound)
    
    Returns:
        PredictionOutput: Sleep quality percentage and reasoning
    """
    try:
        # Calculate sleep quality
        sleep_quality_percent, reasoning = calculate_sleep_quality(
            temperature=input_data.temperature,
            humidity=input_data.humidity,
            light=input_data.light,
            sound=input_data.sound
        )
        
        # Prepare output
        output = {
            "sleep_quality_percent": sleep_quality_percent,
            "reasoning": reasoning
        }
        
        # Log to Firestore
        input_dict = input_data.model_dump()
        log_to_firestore(input_dict, output)
        
        return PredictionOutput(**output)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    firebase_status = "connected" if db is not None else "not configured"
    return {
        "status": "healthy",
        "firebase": firebase_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
