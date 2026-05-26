FROM python:3.12-slim

# gcc diperlukan oleh beberapa dependency (passlib, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies dulu (layer di-cache terpisah dari source code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

# Jalankan sebagai non-root user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Railway menyuntikkan PORT secara otomatis saat runtime
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
