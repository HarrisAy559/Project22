#!/data/data/com.termux/files/usr/bin/bash
# Installation script for Face Recognition Termux

echo "╔══════════════════════════════════════╗"
echo "║  Face Recognition Termux Installer   ║"
echo "╚══════════════════════════════════════╝"

# Update packages
echo "📦 Updating packages..."
pkg update -y && pkg upgrade -y

# Install dependencies
echo "📦 Installing dependencies..."
pkg install -y python git cmake
pkg install -y opencv libjpeg-turbo libpng freetype

# Install Python packages
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install numpy opencv-python-headless Pillow

# Install face-recognition (might take time)
echo "🤖 Installing face-recognition..."
pip install face-recognition

# Grant permissions
echo "🔑 Granting permissions..."
termux-setup-storage
termux-camera-photo test.jpg 2>/dev/null && rm test.jpg

# Create directories
echo "📁 Creating directories..."
mkdir -p dataset known_faces unknown_faces logs models utils scripts

echo ""
echo "✅ Installation complete!"
echo ""
echo "To run the application:"
echo "  python face_recognition.py"
echo ""
echo "To add your first person:"
echo "  1. Run: python face_recognition.py"
echo "  2. Choose option 1: Add New Person"
echo ""
echo "Need help? Check docs/USAGE.md"