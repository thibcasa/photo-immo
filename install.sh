#!/usr/bin/env bash
# Installe tout ce qu'il faut pour le MODE A (Debian/Ubuntu). Idempotent.
set -euo pipefail
cd "$(dirname "$0")"
if command -v apt-get >/dev/null; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libimage-exiftool-perl ffmpeg tesseract-ocr tesseract-ocr-fra fonts-dejavu-core
elif command -v brew >/dev/null; then
  brew install exiftool ffmpeg tesseract tesseract-lang
fi
python3 -m pip install --quiet -r requirements.txt
python3 -m rv2 selfcheck
