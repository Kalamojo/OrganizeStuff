#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Creating temporary build environment ---"
python -m venv /tmp/build-env
source /tmp/build-env/bin/activate

echo "--- Installing build-time dependencies into temp env ---"
pip install -r requirements-build.txt

echo "--- Running model preparation script ---"
# Pass "." as the argument, so the script downloads the model to ./clip_model
python ../scripts/prepare_model.py .

echo "--- Deactivating and cleaning up build environment ---"
deactivate
rm -rf /tmp/build-env

echo "--- Installing runtime dependencies for final package---"
pip install -r requirements.txt

echo "--- Build complete ---"
