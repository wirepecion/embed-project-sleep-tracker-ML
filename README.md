# FastAPI + Firebase + ML Model (Railway Deployment)

Steps:
1. Save Firebase Service Account JSON in GitHub Secrets.
2. Encode it in GitHub Actions:
   B64=$(printf "%s" "$FIREBASE_JSON" | base64 -w 0)
3. GitHub deploy workflow sets FIREBASE_CREDENTIALS_JSON_B64
4. Railway deploys Dockerfile and runs the app.

Prediction Endpoint:
POST /predict
{
  "features": [1.0, 2.0, 3.0]
}
