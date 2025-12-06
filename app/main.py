from fastapi import FastAPI
from app.model_loader import load_model, predict
from app.schemas import PredictRequest, PredictResponse
from app.firebase_client import init_firebase

app = FastAPI()

# init model and firebase
model = load_model()
firebase_app = init_firebase()

@app.get("/")
def root():
    return {"message": "ML API running on Railway 🚀"}

@app.post("/predict", response_model=PredictResponse)
def predict_route(payload: PredictRequest):
    result = predict(payload.features)
    return PredictResponse(prediction=result)
