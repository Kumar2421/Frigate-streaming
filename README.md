# Fusion NVR

A custom NVR (Network Video Recorder) solution based on Frigate, featuring object detection, tracking, and recording capabilities. This project includes custom branding, Docker deployment, and optimized configuration for production use.

## 🌟 Features

- **Object Detection**: YOLOv8-based detection with ONNX runtime
- **Object Tracking**: Multi-object tracking with configurable methods
- **Video Recording**: Automatic event-based recording with retention policies
- **Web Interface**: Modern React-based web UI with custom branding
- **Docker Deployment**: Easy deployment with Docker Compose
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Real-time Processing**: Live camera feeds and real-time detection

## 🚀 Quick Start

### Docker Deployment (Recommended)

**Prerequisites:**
- Docker Desktop installed and running
- At least 4GB RAM available for Docker

**1. Start Docker Desktop** (if not already running)

**2. Build the frontend:**
```bash
cd web
npm install
npm run build:local
cd ..
```

**3. Start Fusion NVR:**
```bash
docker compose up -d
```

**4. Access the web interface:**
- Open your browser and navigate to: `http://localhost:5001`
- The web interface will show "Fusion NVR" with custom branding

**5. Stop Fusion NVR:**
```bash
docker compose down
```

### Python Environment Setup (Alternative)

**For Python 3.11+ (Latest):**
```bash
setup-python-env.bat
```

**For Python 3.10 (Compatible):**
```bash
setup-python310-env.bat
```

**Run Fusion NVR:**
```bash
# For Python 3.11+
.\run-frigate-python.bat

# For Python 3.10
run-frigate-python310.bat
```

### Manual Setup

**1. Create Virtual Environment:**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**2. Install Dependencies:**
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install DeepOCSORT and tracking packages
pip install git+https://github.com/GerardMaggiolino/Deep-OC-SORT.git
pip install torchreid lap cython-bbox

# Install Frigate requirements
pip install -r requirements-python.txt
```

**3. Run Frigate:**
```bash
python -m frigate
```

## 📁 Project Structure

```
frigate/
├── frigate/                 # Main application code
├── web/                     # React web interface (frontend)
│   ├── src/                # React source code
│   ├── dist/               # Built frontend (generated)
│   └── images/             # Custom logo and images
├── config/                 # Configuration files
│   └── config.yml          # Main configuration file
├── input/                  # Video input files
├── docker-compose.yml       # Docker Compose configuration
├── requirements-python.txt # Python dependencies
├── setup-python-env.bat    # Python 3.11+ setup script
├── setup-python310-env.bat # Python 3.10 setup script
├── run-frigate-python.bat  # Run Fusion NVR (Python 3.11+)
└── run-frigate-python310.bat # Run Fusion NVR (Python 3.10)
```

## ⚙️ Configuration

### Basic Configuration

Edit `config/config.yml` to configure cameras, detection, and recording:

```yaml
# MQTT (optional, disabled by default)
mqtt:
  enabled: false

# Global model configuration
model:
  model_type: yolo-generic
  width: 640
  height: 640
  input_tensor: nchw
  input_dtype: float
  input_pixel_format: bgr
  path: "/config/yolo8n.onnx"  # Path to YOLOv8 ONNX model

# Detectors
detectors:
  onnx:
    type: onnx
    device: AUTO  # AUTO, CPU, or GPU (if available)

# Cameras
cameras:
  test_camera:
    ffmpeg:
      inputs:
        - path: input/test_video.mp4
          roles:
            - detect
            - record
    detect:
      width: 1280
      height: 720
      fps: 5
    record:
      enabled: true
      events:
        retain:
          default: 30
          mode: motion
          objects:
            person: 30
    live:
      height: 720
      quality: 8

# Object tracking
track:
  enabled: true
  max_disappeared: 30
  max_distance: 50
```

### Model Setup

**YOLOv8 Model:**
1. Export YOLOv8 to ONNX format:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="onnx", imgsz=[640, 640], opset=20, dynamic=True, dynamo=True)
   ```
2. Place the exported `yolov8n.onnx` file in the `config/` directory
3. Update the `model.path` in `config.yml` to point to the model file

### Docker Configuration

The `docker-compose.yml` file includes:
- Volume mounts for config, recordings, and web interface
- Named volumes for database and model cache (avoids Windows filesystem issues)
- Port mappings for web UI (5001), API (5000), and RTSP (8554, 8555)
- Environment variables for timezone and RTSP password

## 🔧 Requirements

### Docker Deployment
- **Docker Desktop**: Latest version
- **RAM**: 4GB minimum, 8GB recommended for Docker
- **Storage**: 10GB free space
- **OS**: Windows 10/11, Linux, macOS

### Python Deployment (Alternative)
- **Python**: 3.10 or 3.11+ (3.11+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **OS**: Windows 10/11, Linux, macOS

### Python Packages (for Python deployment)
- **PyTorch**: 2.1.0+ (CPU version)
- **OpenCV**: 4.8.0+
- **NumPy**: 1.24.0+
- **ONNX Runtime**: 1.15.0+
- **Frigate**: Latest version

## 🐛 Troubleshooting

### Docker Issues

**1. Docker Desktop not running:**
- Ensure Docker Desktop is started before running `docker compose up -d`
- Check Docker Desktop status in system tray

**2. Port already in use:**
- Change ports in `docker-compose.yml` if 5000, 5001, or 8554 are in use
- Check for other services using these ports

**3. Web interface shows old images:**
- Rebuild the frontend: `cd web && npm run build:local && cd ..`
- Restart Docker: `docker compose down && docker compose up -d`
- Clear browser cache (Ctrl+Shift+R)

**4. Database I/O errors:**
- The Docker setup uses named volumes to avoid Windows filesystem issues
- If issues persist, check Docker volume permissions

**5. Model not found:**
- Ensure `yolo8n.onnx` is in the `config/` directory
- Check the `model.path` in `config.yml` matches the file location

### Python Deployment Issues

**1. Missing packages:**
```bash
pip install -r requirements-python.txt
```

**2. NumPy compatibility issues:**
```bash
pip install "numpy<2.0"
```

**3. ONNX Runtime installation:**
```bash
pip install onnxruntime>=1.15.0
```

**4. PyTorch installation fails:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Python Version Issues

**For Python 3.10:**
- Use `setup-python310-env.bat`
- May need older package versions

**For Python 3.11+:**
- Use `setup-python-env.bat`
- Full feature support

## 📚 Documentation

- **Frigate**: [Official Documentation](https://docs.frigate.video/)
- **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **Docker**: [Docker Documentation](https://docs.docker.com/)
- **ONNX Runtime**: [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Frigate Team**: For the excellent NVR software foundation
- **Ultralytics**: For YOLOv8 object detection models
- **ONNX Runtime Team**: For the efficient inference engine
- **React Community**: For the web framework
- **Docker Team**: For containerization platform

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/your-repo/issues)
3. Create a new issue with detailed information
4. Join the community discussions

---

## 🎨 Customization

### Changing the Logo

1. Replace `web/images/logo.png` with your custom logo (recommended: 512x512px PNG)
2. Rebuild the frontend: `cd web && npm run build:local && cd ..`
3. Restart Docker: `docker compose down && docker compose up -d`

### Changing the Application Name

1. Update translation files in `web/public/locales/en/` (replace "Fusion NVR" with your name)
2. Update `web/site.webmanifest` with your application name
3. Rebuild the frontend and restart Docker

---

**Happy Monitoring! 🎯**