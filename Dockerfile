FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Default env - Railway will override secrets/environment
ENV FIREBASE_CREDENTIALS_PATH=/secrets/serviceAccountKey.json
EXPOSE 8080

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
