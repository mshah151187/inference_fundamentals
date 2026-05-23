#!/bin/bash
# Run this at the start of every new Lambda Labs session.
# Installs everything not pre-installed by Lambda Stack 22.04.

set -e  # stop on first error

echo "Installing Python packages..."
pip install --upgrade Pillow
pip install transformers accelerate

echo "Downloading GPT-2 model (if not already cached)..."
python3 ~/inference_fundamentals/scripts/download_gpt2_model.py

echo "Instance ready."
