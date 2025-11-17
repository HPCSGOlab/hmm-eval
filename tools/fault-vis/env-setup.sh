#!/usr/bin/env bash
set -e

# Name of the virtual environment directory
VENV_DIR=".faultvis-env"

echo "Creating Python virtual environment in: $VENV_DIR"
python3 -m venv $VENV_DIR

echo "Activating environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
pip install plotly pandas

echo ""
echo "✔ Environment setup complete!"
echo "To activate it later, run:"
echo "    source $VENV_DIR/bin/activate"
echo ""

