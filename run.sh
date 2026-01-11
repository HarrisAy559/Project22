#!/data/data/com.termux/files/usr/bin/bash
# Quick start script

echo "Starting Face Recognition..."

# Check if dependencies are installed
if ! python -c "import face_recognition" 2>/dev/null; then
    echo "Installing dependencies..."
    ./setup.sh
fi

# Run the application
python face_recognition.py