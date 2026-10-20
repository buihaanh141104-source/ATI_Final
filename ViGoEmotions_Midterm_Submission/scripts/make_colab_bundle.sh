#!/usr/bin/env bash
# Zip the project for upload to Google Drive (MyDrive/vigoemotions_midterm).
# Includes data/ (PRIVATE: keep the zip on your own Drive, never share publicly).
set -euo pipefail
cd "$(dirname "$0")/.."
out="../vigoemotions_midterm.zip"
rm -f "$out"
zip -qr "$out" . -x ".cache/*" ".venv/*" "*/__pycache__/*" "*.egg-info/*" "results/checkpoints/*" ".DS_Store" ".ipynb_checkpoints/*"
echo "wrote $(cd .. && pwd)/vigoemotions_midterm.zip  ->  unzip into MyDrive/vigoemotions_midterm/"
