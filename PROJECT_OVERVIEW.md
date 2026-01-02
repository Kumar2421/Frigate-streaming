# Frigate NVR with DeepSORT Tracking - Project Overview

## 📋 Project Overview

This is a **customized Frigate NVR (Network Video Recorder)** system with **DeepSORT object tracking** integration. Frigate is an open-source NVR designed for Home Assistant with AI-powered real-time object detection.

### Key Features

- **Real-time Object Detection**: Uses TensorFlow/OpenVINO for local AI object detection
- **DeepSORT Tracking**: Enhanced multi-object tracking with appearance-based re-identification
- **Multi-Camera Support**: Currently configured with 4 cameras (em-dept-exit, em-dept-entry, em-dept-rightexit, em-dept-sideexit)
- **24/7 Recording**: Continuous video recording with object-based retention
- **MQTT Integration**: Communicates via MQTT for Home Assistant integration
- **RTSP Streaming**: Re-streams camera feeds to reduce connections
- **WebRTC Support**: Low-latency live view
- **Hardware Acceleration**: Configured for Intel GPU (renderD128) and USB Coral AI accelerator

### Architecture

```
┌─────────────────┐
│   IP Cameras    │ (4 cameras via RTSP)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frigate NVR    │ (Custom container with DeepSORT)
│  - Detection    │
│  - Tracking     │
│  - Recording    │
└────────┬────────┘
         │
         ├──► MQTT Broker ──► Home Assistant
         │
         └──► Media Storage (/mnt/additional-disk/frigate/media)
```

### Custom Components

1. **Custom Detector Container** (`custom-detector/`)
   - Base: Frigate stable image
   - Enhanced with: DeepSORT tracking library
   - Custom tracker selector for switching between Norfair and DeepSORT

2. **DeepSORT Tracker** (`frigate/track/deepsort_tracker.py`)
   - Implements DeepSORT algorithm for better object tracking
   - Configurable via environment variables

3. **Tracker Selector** (`frigate/track/tracker_selector.py`)
   - Dynamically selects tracking algorithm based on `USE_DEEPSORT` env var

## 🚀 How to Run

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM recommended
- GPU or AI accelerator (Intel GPU/Google Coral) for better performance
- Network access to IP cameras

### Quick Start

1. **Navigate to project directory:**
   ```bash
   cd /mnt/additional-disk/frigate
   ```

2. **Build and start services:**
   ```bash
   docker compose up -d --build
   ```

3. **View logs:**
   ```bash
   # All services
   docker compose logs -f
   
   # Frigate only
   docker compose logs -f frigate
   
   # MQTT only
   docker compose logs -f mqtt
   ```

4. **Access Frigate Web UI:**
   - Open browser: `http://localhost:5000`
   - Default login: (check config for authentication)

5. **Stop services:**
   ```bash
   docker compose down
   ```

### Configuration

**Main Config**: `/mnt/additional-disk/frigate/config/config.yml`
- Camera settings
- Detection zones
- Recording settings
- Object filters

**Environment Variables** (in `docker-compose.yml`):
- `USE_DEEPSORT=true` - Enable DeepSORT tracking
- `DEEPSORT_MAX_AGE=30` - Max frames to keep lost tracks
- `DEEPSORT_MAX_COSINE_DISTANCE=0.2` - Appearance matching threshold
- `DEEPSORT_MAX_IOU_DISTANCE=0.7` - IoU matching threshold
- `DEEPSORT_N_INIT=3` - Frames before confirming track
- `DEEPSORT_NN_BUDGET=100` - Appearance descriptor budget

### Rebuilding After Changes

```bash
# Rebuild custom detector
docker compose build frigate

# Restart with new build
docker compose up -d frigate
```

## 💾 Disk Space Analysis

### Current Disk Usage

| Mount Point | Size | Used | Available | Usage % |
|------------|------|------|-----------|---------|
| **Root (`/`)** | 196G | 187G | 7.2G | **97%** ⚠️ |
| **Additional Disk (`/mnt/additional-disk`)** | 74G | 47G | 24G | 67% ✅ |

### Project Directory Usage

```
/mnt/additional-disk/frigate/
├── config/         354M  (camera configs, recordings metadata)
├── media/          128K  (video recordings - will grow)
├── mqtt/           460K  (MQTT broker data)
├── frigate/        2.6M  (source code)
├── web/            11M   (web UI assets)
├── docs/           53M   (documentation)
└── docker/         4.0M  (Docker files)
```

### Docker Space Usage

```
Images:      59.68GB total (18 active, 41.61GB reclaimable)
Containers:  800.2MB
Volumes:     1.269GB
Build Cache: 19.1GB (can be cleaned)
```

### Recommendations

1. **⚠️ CRITICAL: Root partition is 97% full!**
   - Clean up old Docker images: `docker system prune -a`
   - Move Docker data to additional disk (see below)
   - Clean build cache: `docker builder prune`

2. **Use Additional Disk for Docker:**
   - `/mnt/additional-disk` has 24GB free
   - Better performance and more space

3. **Monitor Media Growth:**
   - Current media folder: 128K (newly created)
   - Will grow with recordings
   - Configure retention in `config.yml`

## 🔧 Docker Mount Recommendations

### Current Mounts

```yaml
volumes:
  - /mnt/additional-disk/frigate/config:/config
  - /mnt/additional-disk/frigate/media:/media/frigate
  - type: tmpfs
    target: /tmp/cache
    tmpfs:
      size: 1000000000  # ~1GB RAM cache
```

### Recommended Additional Mounts

1. **Move Docker Root to Additional Disk** (if needed):
   ```bash
   # Stop Docker
   sudo systemctl stop docker
   
   # Move Docker data
   sudo mv /var/lib/docker /mnt/additional-disk/docker
   
   # Create symlink
   sudo ln -s /mnt/additional-disk/docker /var/lib/docker
   
   # Start Docker
   sudo systemctl start docker
   ```

2. **Add Docker Compose Override** (optional):
   ```yaml
   # docker-compose.override.yml
   services:
     frigate:
       volumes:
         - /mnt/additional-disk/frigate/clips:/media/frigate/clips
         - /mnt/additional-disk/frigate/recordings:/media/frigate/recordings
         - /mnt/additional-disk/frigate/snapshots:/media/frigate/snapshots
   ```

3. **Available Mount Points:**
   - ✅ `/mnt/additional-disk` - **24GB free** (RECOMMENDED)
   - ⚠️ `/` (root) - **7.2GB free** (CRITICAL - avoid if possible)

### Space Optimization Commands

```bash
# Clean Docker system (reclaim ~60GB)
docker system prune -a --volumes

# Clean build cache (reclaim ~19GB)
docker builder prune -a

# Check what's using space
docker system df -v

# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune
```

## 📊 System Resources

### Current Setup

- **CPU**: Multi-core (check with `nproc`)
- **RAM**: 31GB available (from `/dev/shm`)
- **GPU**: Intel GPU (renderD128) for hardware acceleration
- **AI Accelerator**: USB Coral device (if connected)

### Performance Tips

1. **Use Hardware Acceleration:**
   - Intel GPU: Already configured (`/dev/dri/renderD128`)
   - Google Coral: USB device passed through

2. **Optimize Detection:**
   - Reduce detection width/height if needed
   - Use motion detection zones
   - Adjust detection FPS per camera

3. **Storage:**
   - Use additional disk for recordings
   - Configure retention policies
   - Consider external storage for long-term archives

## 🔍 Troubleshooting

### Check Container Status
```bash
docker compose ps
```

### View Real-time Logs
```bash
docker compose logs -f frigate
```

### Access Container Shell
```bash
docker exec -it frigate bash
```

### Check DeepSORT Status
```bash
docker exec frigate env | grep DEEPSORT
```

### Verify Camera Feeds
```bash
# Test RTSP stream
ffmpeg -i rtsp://admin:Admin@123@121.200.48.187:554/Streaming/Channels/102 -t 5 test.mp4
```

## 📝 Next Steps

1. **Immediate Actions:**
   - ✅ Clean Docker to free up root partition space
   - ✅ Monitor media folder growth
   - ✅ Verify all cameras are working

2. **Optimization:**
   - Configure retention policies
   - Set up Home Assistant integration
   - Fine-tune DeepSORT parameters

3. **Monitoring:**
   - Set up disk space alerts
   - Monitor CPU/GPU usage
   - Review detection accuracy

## 🔗 Useful Links

- **Frigate Docs**: https://docs.frigate.video
- **GitHub**: https://github.com/blakeblackshear/frigate
- **Home Assistant Integration**: https://github.com/blakeblackshear/frigate-hass-integration

---

**Last Updated**: 2025-09-12
**Project Location**: `/mnt/additional-disk/frigate`

