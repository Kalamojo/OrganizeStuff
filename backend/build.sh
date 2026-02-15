#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing build-time dependencies ---"
pip install -r requirements-build.txt

echo "--- Running model preparation script ---"
# Pass "." as the argument, so the script downloads the model to ./clip_model
python prepare_model.py .

echo "--- Installing runtime dependencies ---"
pip install -r requirements.txt

echo "--- Build complete ---"
