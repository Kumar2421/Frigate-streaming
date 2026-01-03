@echo off
echo ========================================
echo Python 3.10 Compatible Environment Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python is installed ✓
python --version
echo.

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Detected Python version: %PYTHON_VERSION%
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo pip is available ✓
echo.

REM Create virtual environment
echo Creating Python virtual environment...
if exist "venv310" (
    echo Virtual environment already exists, removing old one...
    rmdir /s /q venv310
)

python -m venv venv310
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created ✓
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv310\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated ✓
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch first (CPU version)
echo Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo PyTorch installed ✓
echo.

REM Install basic requirements
echo Installing basic requirements...
pip install opencv-python numpy scipy scikit-learn pillow flask requests websockets peewee pyyaml psutil
if %errorlevel% neq 0 (
    echo ERROR: Failed to install basic requirements
    pause
    exit /b 1
)
echo Basic requirements installed ✓
echo.

REM Install DeepOCSORT
echo Installing DeepOCSORT...
pip install git+https://github.com/GerardMaggiolino/Deep-OC-SORT.git
if %errorlevel% neq 0 (
    echo WARNING: DeepOCSORT installation failed, continuing...
) else (
    echo DeepOCSORT installed ✓
)
echo.

REM Install TorchReID
echo Installing TorchReID...
pip install torchreid
if %errorlevel% neq 0 (
    echo WARNING: TorchReID installation failed, continuing...
) else (
    echo TorchReID installed ✓
)
echo.

REM Install additional packages
echo Installing additional packages...
pip install lap cython-bbox gdown sherpa-onnx librosa faster-whisper
if %errorlevel% neq 0 (
    echo WARNING: Some additional packages failed to install, continuing...
) else (
    echo Additional packages installed ✓
)
echo.

REM Install older Frigate version that works with Python 3.10
echo Installing Frigate (Python 3.10 compatible version)...
pip install "frigate<0.15.0"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Frigate
    echo Trying alternative installation...
    pip install frigate==0.14.1
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install any Frigate version
        pause
        exit /b 1
    )
)
echo Frigate installed ✓
echo.

REM Create necessary directories
echo Creating directories...
if not exist "config" mkdir config
if not exist "input\recordings" mkdir input\recordings
if not exist "input\clips" mkdir input\clips
if not exist "cache" mkdir cache
echo Directories created ✓
echo.

REM Create config file if it doesn't exist
if not exist "config\config.yml" (
    echo Creating default config file...
    if exist "config\config-clean.yml" (
        copy "config\config-clean.yml" "config\config.yml"
        echo Config file created from clean template ✓
    ) else (
        echo Creating basic config file...
        echo # Frigate Configuration > "config\config.yml"
        echo mqtt: >> "config\config.yml"
        echo   enabled: false >> "config\config.yml"
        echo cameras: {} >> "config\config.yml"
        echo Basic config file created ✓
    )
) else (
    echo Config file already exists ✓
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Python 3.10 compatible environment is ready!
echo.
echo To activate the environment in the future, run:
echo   venv310\Scripts\activate.bat
echo.
echo To run Frigate, use: run-frigate-python310.bat
echo.
echo NOTE: This uses an older Frigate version for Python 3.10 compatibility.
echo For full features, consider upgrading to Python 3.11+.
echo.
echo Press any key to exit...
pause >nul
