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
├── example.py           # Example usage script
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
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Required (Core)
- [ ] `POST /v1/score/interval` — score a single 5-minute interval
- [ ] `POST /v1/score/interval/batch` — score many intervals in one request (gateway-friendly)
- [ ] `POST /v1/score/session` — score an entire sleep session (aggregate)
- [ ] `GET /v1/health` — liveness / basic health check
- [ ] `GET /v1/model/info` — current model metadata (version, features, ranges)

### Optional / Ops
- [ ] `POST /v1/model/reload` — hot-reload model artifact (admin)
- [ ] `GET /v1/model/versions` — list available model artifacts/versions (admin)
- [ ] `POST /v1/debug/score` — verbose debug scoring (dev-only, protected)

## License

MIT License