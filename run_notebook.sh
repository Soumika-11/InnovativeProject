#!/bin/bash
# Face Verification System - Quick Launch Script

echo "🚀 Face Verification System - Starting..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "✅ Virtual environment created!"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed!"
else
    echo "✅ Dependencies already installed!"
fi

# Launch Jupyter Notebook
echo ""
echo "📓 Launching Jupyter Notebook..."
echo "💡 The notebook will open in your default browser"
echo "💡 Press Ctrl+C to stop the server when done"
echo ""
jupyter notebook face_verification_complete.ipynb
