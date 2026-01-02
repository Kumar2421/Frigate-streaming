@echo off
REM Build custom Frigate Docker image with DeepOCSORT support

echo 🌟 Building custom Frigate Docker image with DeepOCSORT support...

REM Build the custom image
echo 📦 Building Docker image...
docker build -f Dockerfile.deepocsort -t frigate-deepocsort:latest .

if errorlevel 1 (
    echo ❌ Failed to build Docker image
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Create config directory if it doesn't exist
if not exist "config" (
    echo 📁 Creating config directory...
    mkdir config
)

REM Copy example config if config.yml doesn't exist
if not exist "config\config.yml" (
    echo 📋 Copying example configuration...
    copy "config\config.yml.example" "config\config.yml"
)

echo.
echo 🚀 Setup complete! You can now run Frigate with DeepOCSORT using:
echo.
echo    docker-compose -f docker-compose.deepocsort.yml up -d
echo.
echo Or run directly with:
echo    docker run --rm --publish=5000:5000 --volume="%CD%/config:/config" frigate-deepocsort:latest
echo.
pause
