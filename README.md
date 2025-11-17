# Sleep Tracker ML

A FastAPI backend that predicts sleep quality based on environmental factors including temperature, humidity, light, and sound levels.

## Features

- **POST /predict**: Predict sleep quality percentage from environmental data
- **Rule-based model**: Simple placeholder for future ML models
- **Firebase Integration**: Logs all predictions to Firestore (when configured)
- **Input Validation**: Robust validation using Pydantic models
- **Interactive Documentation**: Auto-generated API docs at `/docs`

## Project Structure

```
sleep_tracker_ML/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wirepecion/sleep_tracker_ML.git
cd sleep_tracker_ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure Firebase:
   - Download your Firebase service account key from Firebase Console
   - Save it as `serviceAccountKey.json` in the project root
   - Or set environment variable: `export FIREBASE_CREDENTIALS_PATH=/path/to/key.json`

## Running the API

Start the server:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Root endpoint with API information

### GET /health
Health check endpoint showing API and Firebase status

### POST /predict
Predict sleep quality from environmental factors

**Request Body:**
```json
{
  "temperature": 20.0,
  "humidity": 40.0,
  "light": 5.0,
  "sound": 30.0
}
```

**Parameters:**
- `temperature`: Temperature in Celsius (-50 to 60°C)
- `humidity`: Humidity percentage (0-100%)
- `light`: Light level in lux (≥0)
- `sound`: Sound level in decibels (0-200 dB)

**Response:**
```json
{
  "sleep_quality_percent": 100.0,
  "reasoning": "Temperature is optimal. Humidity is optimal. Light level is optimal (dark). Sound level is optimal (quiet)."
}
```

## Sleep Quality Model

The current implementation uses a rule-based model with the following optimal conditions:

- **Temperature**: 15-22°C (59-72°F)
- **Humidity**: 30-50%
- **Light**: 0-10 lux (dark room)
- **Sound**: 0-40 dB (quiet)

The model calculates a score from 0-100% based on how far the conditions deviate from optimal values.

## Example Usage

### Using curl:

Optimal conditions (100% quality):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 20, "humidity": 40, "light": 5, "sound": 30}'
```

Poor conditions (low quality):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 30, "humidity": 70, "light": 300, "sound": 85}'
```

### Using Python:

```python
import requests

data = {
    "temperature": 18,
    "humidity": 45,
    "light": 25,
    "sound": 50
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Sleep Quality: {result['sleep_quality_percent']}%")
print(f"Reasoning: {result['reasoning']}")
```

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Firebase Logging

When Firebase is configured, each prediction is logged to Firestore with:
- Input parameters (temperature, humidity, light, sound)
- Output (sleep_quality_percent, reasoning)
- Timestamp

The logs are stored in the `predictions` collection.

## Future Improvements

- Replace rule-based model with ML model (scikit-learn, TensorFlow, PyTorch)
- Add historical data analysis
- Implement user authentication
- Add batch prediction endpoint
- Store and retrieve user sleep patterns
- Add more environmental factors (air quality, CO2 levels, etc.)

## License

MIT License