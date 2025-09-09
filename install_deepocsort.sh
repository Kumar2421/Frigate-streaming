#!/bin/bash

# DeepOCSORT Installation Script for Frigate
# This script installs the required dependencies for DeepOCSORT with YOLO re-identification

echo "🌟 Installing DeepOCSORT with YOLO re-identification for Frigate..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version is not supported. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Install PyTorch (CPU version by default)
echo "📦 Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📦 Installing DeepOCSORT and ReID dependencies..."
pip3 install -r requirements-deepocsort.txt

# Download ReID model if not present
echo "📥 Downloading ReID model..."
python3 -c "
try:
    import torchreid
    model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
    print('✅ ReID model downloaded successfully')
except Exception as e:
    print(f'❌ Failed to download ReID model: {e}')
    print('Note: ReID models will be downloaded automatically on first use')
"

# Test installation
echo "🧪 Testing installation..."
python3 -c "
try:
    import deepocsort
    import torchreid
    import torch
    import cv2
    print('✅ All dependencies installed successfully!')
    print('🎉 DeepOCSORT with ReID re-identification is ready to use!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit 1
"

echo ""
echo "🚀 Installation complete! You can now:"
echo "   1. Update your Frigate config to use 'deepocsort' tracker"
echo "   2. Configure re-identification parameters in the web interface"
echo "   3. Restart Frigate to apply the new tracker"
echo ""
echo "📖 For more information, see the Frigate documentation."
