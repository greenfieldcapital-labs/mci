#!/usr/bin/env bash
set -e  # Exit on error

echo "🔧 Setting up environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.13..."
    uv venv -p python3.13
else
    echo "Virtual environment already exists, reusing..."
fi

# Activate virtual environment
source .venv/bin/activate

# Install the package in editable mode along with dependencies
echo "📦 Installing dependencies..."
uv pip install -e ".[test]"
uv pip install jupyterlab

# Start Jupyter Lab
echo "🚀 Starting Jupyter Lab..."
echo "Opening example.ipynb..."
jupyter lab example.ipynb

