#!/usr/bin/env bash
set -euo pipefail

echo "Starting Flokka Ingestion Engine..."

# Create upload directory if it doesn't exist
mkdir -p uploads

# Start FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
