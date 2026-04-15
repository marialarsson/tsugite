#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
cd "$DIR/setup"
python Tsugite_app.py
