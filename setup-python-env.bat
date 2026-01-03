@echo off
echo ========================================
echo Python Environment Setup for Frigate
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python is installed ✓
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo pip is available ✓
echo.

REM Create virtual environment
echo Creating Python virtual environment...
if exist "venv" (
    echo Virtual environment already exists, removing old one...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created ✓
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
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

REM Install Windows-compatible requirements
echo Installing Windows-compatible Frigate requirements...
pip install -r requirements-windows.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install, trying minimal requirements...
    pip install -r requirements-minimal.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install requirements
        echo.
        echo TIP: If you get build errors, you may need to install:
        echo - Microsoft Visual C++ Build Tools
        echo - Or use pre-compiled wheels
        pause
        exit /b 1
    )
    echo Minimal requirements installed ✓
) else (
    echo Windows-compatible requirements installed ✓
)
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
    if exist "config\config.yml.example" (
        copy "config\config.yml.example" "config\config.yml"
        echo Config file created from example ✓
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
echo Python environment is ready with Frigate and DeepOCSORT!
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run Frigate, use: run-frigate-python.bat
echo.
echo Press any key to exit...
pause >nul