@echo off
echo ========================================
echo Running Frigate with Python Environment
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo ERROR: Virtual environment not found
    echo Please run setup-python-env.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating Python virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated ✓
echo.

REM Check if config exists
if not exist "config\config.yml" (
    echo ERROR: Config file not found
    echo Please ensure config\config.yml exists
    pause
    exit /b 1
)
echo Config file found ✓
echo.

REM Set environment variables
set FRIGATE_CONFIG_DIR=%cd%\config
set FRIGATE_RECORDINGS_DIR=%cd%\input\recordings
set FRIGATE_CLIPS_DIR=%cd%\input\clips
set FRIGATE_CACHE_DIR=%cd%\cache

echo Environment variables set:
echo   Config: %FRIGATE_CONFIG_DIR%
echo   Recordings: %FRIGATE_RECORDINGS_DIR%
echo   Clips: %FRIGATE_CLIPS_DIR%
echo   Cache: %FRIGATE_CACHE_DIR%
echo.

echo Starting Frigate...
echo Web interface will be available at: http://localhost:5001
echo Press Ctrl+C to stop Frigate
echo.

REM Run Frigate
python -m frigate
