#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- VERCEL BUILD SCRIPT START ---"

# 1. Build Frontend
echo "--- Building frontend ---"
# Navigate to the frontend directory, install dependencies, and build
cd frontend
npm install
npm run build
cd .. # Go back to the project root

# 2. Prepare Backend Assets (Models and Tokenizer)
echo "--- Preparing backend assets ---"
# Install the heavy, build-time-only dependencies
pip install -r backend/requirements-build.txt
# Run the preparation script. It will place models/tokenizer inside the 'backend' folder.
python scripts/prepare_model.py backend

echo "--- VERCEL BUILD SCRIPT COMPLETE ---"
