# 🌙 Neuro-Symbolic Sleep Environment Tracker (IoT + AI)

> **A Knowledge-Informed AI System that optimizes sleep environments in real-time.**
>
> *Deployed on Railway | Powered by FastAPI, Firebase, & Scikit-Learn*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?logo=fastapi)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-orange?logo=firebase)
![Scikit-Learn](https://img.shields.io/badge/AI-RandomForest-yellow?logo=scikit-learn)
![Status](https://img.shields.io/badge/System-Production%20Ready-brightgreen)

---

## 📖 Overview
This repository hosts the **Backend Intelligence Service** for an IoT Sleep Tracking System tailored for the tropical climate of **Thailand**.

Unlike traditional "Black Box" AI models, this system uses a **Hybrid Neuro-Symbolic Architecture** ($Score = Rule + Model$). It combines deterministic "Physics Rules" (localized for Thai heat/humidity acclimatization) with a "Residual ML Model" that learns complex non-linear interactions (e.g., the compounding stress of Heat + Noise).

### 🚀 Key Capabilities
* **Real-time Scoring:** Processes sensor data (Temp, Humidity, Light, Sound) every 5 minutes using a micro-batching architecture.
* **IoT Automation:** Automatically triggers a **Blynk-connected Diffuser** if sleep quality drops below 50%.
* **Cost-Optimized Architecture:** Uses "Time-Bounded Polling" to minimize database reads (reducing daily reads from ~33k to ~300).
* **Smart Notifications:** Sends a comprehensive HTML Summary Report via Email (Resend/Gmail) immediately upon wake-up.
* **Resilient Design:** Handles "Zombie Sessions" (auto-close >24h), "Cold Starts", and network failures gracefully.

---

## 🏗️ Architecture

The system follows an **Event-Driven Micro-Batching** pattern to handle asynchronous IoT data efficiently.

```mermaid
graph TD
    A[ESP32 IoT Node] -->|Writes Data (5 min)| B(Firebase Firestore)
    C[Railway Backend] -->|Polls Active Sessions| B
    B -->|New Data found| C
    C -->|1. Compute Physics Rule| D[Rule Engine]
    C -->|2. Predict Residual Error| E[Random Forest Model]
    D & E -->|3. Hybrid Score = Rule + Model| F[Final Score]
    F -->|Write Score| B
    F -->|Score < 50%?| G{Trigger IoT?}
    G -->|Yes| H[Call Blynk API -> Diffuser ON]
    I[User Wakes Up] -->|End Session| B
    B -->|Session END Detected| C
    C -->|Generate Summary| J[Email Report (Resend API)]
````

### 🧠 The "Hybrid" Logic (Why it works)

Pure AI fails in Thailand because generic datasets don't account for local heat tolerance.

  * **The Rule Base (`sleep_rules.py`):** Enforces hard limits (e.g., Light \> 50 lux is always bad). Adjusted for Thai thermal comfort (Ideal \~26°C-28°C).
  * **The AI Model (`model.joblib`):** Trained on the **Residuals** (Errors). It learns *only* what the rules miss, making it highly efficient and explainable.

-----

## 📂 Project Structure

```bash
.
├── app/
│   ├── main.py            # Entry point (FastAPI + Background Poller)
│   ├── services.py        # Core Logic (Active Scan, Aggregation, Email, IoT Trigger)
│   ├── sleep_rules.py     # Deterministic Physics Logic (Shared Truth)
│   ├── model_loader.py    # Singleton Model Loader (RAM Optimized)
│   └── firebase_client.py # Database Connection Factory
├── model/
│   └── model.joblib       # Pre-trained Random Forest Residual Model
├── scripts/
│   ├── simulate_hardware.py # IoT Simulator for testing
│   └── train_model.py     # Training pipeline (CSV -> Residual Model)
└── tests/                 # Integration tests
```

-----

## 🛠️ Setup & Deployment

### 1\. Prerequisites

  * Python 3.10+
  * Firebase Project (Firestore enabled)
  * Blynk Account (for IoT control)
  * Resend API Key (for Emails)

### 2\. Environment Variables

Create a `.env` file or configure Railway Variables:

```ini
# Database (Base64 encoded Service Account JSON)
FIREBASE_CREDENTIALS_JSON_B64="ey..."

# Email (Resend API is recommended for Cloud)
RESEND_API_KEY="re_123..."
GMAIL_RECEIVER="user@example.com"

# IoT Config
BLYNK_AUTH_TOKEN="your_token"
```

### 3\. Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (Poller auto-starts)
uvicorn app.main:app --reload

# In a separate terminal, run the Hardware Simulator
python scripts/simulate_hardware.py
```

-----

## 📊 Data Schema (Firestore)

The system manages the full lifecycle of a sleep session:

| Collection | Document | Description |
| :--- | :--- | :--- |
| `sleep_sessions` | `{ "type": "START", "timestamp": "..." }` | The "Anchor" document. Updates to "END" on wake-up. |
| `sensor_readings`| `{ "temp": 26.5, "is_processed": false }` | Raw IoT stream. Backend marks `true` after scoring. |
| `interval_reports`| `{ "score": 85.2, "base": 80, "ai": 5.2 }` | The calculated Hybrid Score (Every 5 mins). |
| `session_records`| `{ "avgTemp": 26.1, "finalScore": 91.2 }` | Final Summary generated upon wake-up. |

-----

## 🧪 Testing

The repo includes a robust test suite that works **offline** (mocking Firebase) or **online** (Integration test).

```bash
# Run safe offline logic test
python -m unittest tests/test_flow_local.py

# Run real database integration test (Requires Env Vars)
python tests/test_real_db.py
```
