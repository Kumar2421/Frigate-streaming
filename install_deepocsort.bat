@echo off
REM DeepOCSORT Installation Script for Frigate (Windows)
REM This script installs the required dependencies for DeepOCSORT with YOLO re-identification

echo 🌟 Installing DeepOCSORT with YOLO re-identification for Frigate...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Install PyTorch (CPU version by default)
echo 📦 Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo 📦 Installing DeepOCSORT and ReID dependencies...
pip install -r requirements-deepocsort.txt

REM Download ReID model if not present
echo 📥 Downloading ReID model...
python -c "import torchreid; model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True); print('✅ ReID model downloaded successfully')"

REM Test installation
echo 🧪 Testing installation...
python -c "import deepocsort; import torchreid; import torch; import cv2; print('✅ All dependencies installed successfully!'); print('🎉 DeepOCSORT with ReID re-identification is ready to use!')"

echo.
echo 🚀 Installation complete! You can now:
echo    1. Update your Frigate config to use 'deepocsort' tracker
echo    2. Configure re-identification parameters in the web interface
echo    3. Restart Frigate to apply the new tracker
echo.
echo 📖 For more information, see the Frigate documentation.
pause
