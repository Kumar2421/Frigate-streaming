# 🎥 Frigate Beginner's Guide - Complete Setup & Usage

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [First Time Access](#first-time-access)
5. [Adding Camera Streams](#adding-camera-streams)
6. [Starting & Stopping Frigate](#starting--stopping-frigate)
7. [Web Interface Overview](#web-interface-overview)
8. [Basic Configuration](#basic-configuration)
9. [Troubleshooting](#troubleshooting)
10. [Common Tasks](#common-tasks)

---



### What You'll Learn

- How to install and run Frigate
- How to add your camera streams
- How to start and stop the service
- How to access the web interface
- Basic configuration and troubleshooting

---

## 📦 Prerequisites

Before you begin, make sure you have:

- **Ubuntu/Linux server** (or any Linux distribution)
- **Docker** installed and running
- **Docker Compose** installed
- **At least 4GB RAM** (8GB+ recommended)
- **Camera with RTSP stream** (or test stream URL)
- **Basic terminal/command line knowledge**

### Check Prerequisites

```bash
# Check Docker is installed
docker --version
# Should show: Docker version 20.x.x or higher

# Check Docker Compose is installed
docker compose version
# Should show: Docker Compose version v2.x.x

# Check Docker is running
docker ps
# Should show running containers (or empty list, both are fine)
```

---

## 🚀 Installation & Setup

### Step 1: Create Project Directory

```bash
# Create a directory for Frigate
mkdir -p ~/frigate
cd ~/frigate

# Create subdirectories for config and media
mkdir -p config input
```

### Step 2: Create docker-compose.yml

Create the `docker-compose.yml` file in the `~/frigate` directory:

```bash
cd ~/frigate
nano docker-compose.yml
```

Paste the following content:

```yaml
services:
  frigate:
    container_name: frigate
    restart: unless-stopped
    stop_grace_period: 30s
    image: kumar2421/frigate-custom:v1.1.4
    shm_size: '512mb'

    volumes:
      - ./config:/config:rw
      - frigate_db:/db
      - ./input:/media/frigate:rw
      - type: tmpfs
        target: /tmp/cache
        tmpfs:
          size: 1000000000
      - frigate_model_cache:/config/model_cache

    ports:
      - "8971:8971"   # Web UI (HTTPS)
      - "5000:5000"   # Internal API (HTTP)
      - "5001:5001"   # Internal API (HTTP, authenticated)
      - "8554:8554"   # RTSP feeds
      - "8555:8555/tcp" # WebRTC over TCP
      - "8555:8555/udp" # WebRTC over UDP

    environment:
      - TZ=Asia/Kolkata
      - FRIGATE_RTSP_PASSWORD=password
      - FRIGATE_CONFIG_DIR=/config

volumes:
  frigate_model_cache:
  frigate_db:
```

**Save the file:**
- In **nano**: Press `Ctrl+O`, then `Enter`, then `Ctrl+X`
- In **vim**: Press `Esc`, type `:wq`, then `Enter`

### Step 3: Pull the Docker Image

```bash
# Login to Docker Hub (if not already logged in)
docker login

# Pull the Frigate image
docker pull kumar2421/frigate-custom:v1.1.4

# This will take 10-20 minutes (image is ~9GB)
# You'll see progress like:
# v1.1.3: Pulling from kumar2421/frigate-custom
# ...
```

### Step 4: Verify Setup

```bash
# Check docker-compose.yml is valid
docker compose config

# Should show your configuration without errors
```

---

## 🌐 First Time Access

### Step 1: Start Frigate

```bash
cd ~/frigate

# Start Frigate in the background
docker compose up -d

# Check status
docker compose ps

# Should show:
# NAME      STATUS
# frigate   Up (healthy)
```

### Step 2: Wait for Startup

Frigate needs 30-60 seconds to fully start. Check logs:

```bash
# Watch the logs
docker compose logs -f frigate

# Wait until you see:
# [INFO] Starting Frigate...
# [INFO] Service Frigate started successfully
```

Press `Ctrl+C` to stop watching logs.

### Step 3: Access Web Interface

Open your web browser and navigate to:

```
https://localhost:8971
```

**Or if accessing from another computer:**

```
https://YOUR_SERVER_IP:8971
```

**Important Notes:**
- Use **HTTPS** (not HTTP) for port 8971
- You may see a **certificate warning** - click "Advanced" → "Proceed anyway"
- This is normal (self-signed certificate)

### Step 4: First Login

On first startup, Frigate generates an admin password. Find it in the logs:

```bash
# Check for admin credentials
docker logs frigate 2>&1 | grep -i "admin\|password\|login"

# Or view all recent logs
docker logs frigate 2>&1 | tail -50
```

**If you don't see the password**, you can reset it:

1. Edit `config/config.yml`:
```bash
nano ~/frigate/config/config.yml
```

2. Add this section:
```yaml
auth:
  reset_admin_password: true
```

3. Restart Frigate:
```bash
docker compose restart frigate
```

4. Check logs again for the new password:
```bash
docker logs frigate 2>&1 | grep -i "admin\|password"
```

5. **Remove the reset line** after first login for security:
```yaml
# Remove or comment out:
# auth:
#   reset_admin_password: true
```

---

## 📹 Adding Camera Streams

### Understanding Camera Configuration

Frigate needs to know where your camera streams are. Most cameras support **RTSP** (Real-Time Streaming Protocol).

### Step 1: Find Your Camera's RTSP URL

Common RTSP URL formats:

**Hikvision:**
```
rtsp://username:password@camera_ip:554/Streaming/Channels/101
```

**Dahua:**
```
rtsp://username:password@camera_ip:554/cam/realmonitor?channel=1&subtype=0
```

**Reolink:**
```
rtsp://username:password@camera_ip:554/h264Preview_01_main
```

**Generic/ONVIF:**
```
rtsp://username:password@camera_ip:554/stream1
```

**Test Stream (for testing):**
```
rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4
```

### Step 2: Edit Configuration File

```bash
# Open the config file
nano ~/frigate/config/config.yml
```

### Step 3: Add Camera Configuration

Find the `cameras:` section and add your camera. Here's a complete example:

```yaml
cameras:
  # Example: Front Door Camera
  front_door:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/101
          roles:
            - detect
            - record
    detect:
      width: 1920
      height: 1080
      fps: 5
    record:
      enabled: true
      retain:
        days: 7
    snapshots:
      enabled: true
      retain:
        days: 30
    motion:
      mask:
        - 0,0,100,100  # Example mask coordinates

  # Example: Backyard Camera
  backyard:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://admin:password123@192.168.1.101:554/Streaming/Channels/101
          roles:
            - detect
            - record
    detect:
      width: 1920
      height: 1080
      fps: 5
    record:
      enabled: true
      retain:
        days: 7
    snapshots:
      enabled: true
      retain:
        days: 30
```

### Step 4: Configuration Explained

**Basic Camera Settings:**

```yaml
camera_name:           # Unique name (no spaces, use underscores)
  enabled: true        # Enable/disable this camera
  ffmpeg:
    inputs:
      - path: rtsp://...  # Your camera's RTSP URL
        roles:
          - detect      # Use for object detection
          - record      # Record video clips
          - rtmp        # RTMP stream (optional)
  detect:
    width: 1920        # Camera resolution width
    height: 1080       # Camera resolution height
    fps: 5             # Frames per second for detection (lower = less CPU)
  record:
    enabled: true      # Enable recording
    retain:
      days: 7          # Keep recordings for 7 days
  snapshots:
    enabled: true      # Enable snapshots
    retain:
      days: 30         # Keep snapshots for 30 days
```

### Step 5: Save and Restart

```bash
# Save the config file (Ctrl+O, Enter, Ctrl+X in nano)

# Validate the configuration
docker compose exec frigate python3 -c "from frigate.config import FrigateConfig; FrigateConfig.load()"
# Should show no errors if config is valid

# Restart Frigate to apply changes
docker compose restart frigate

# Watch logs to see if camera connects
docker compose logs -f frigate
```

### Step 6: Verify Camera is Working

1. **Check logs** for connection status:
```bash
docker logs frigate 2>&1 | grep -i "camera_name\|ffmpeg\|stream"
```

2. **Check web interface:**
   - Go to `https://localhost:8971`
   - You should see your camera in the camera list
   - Click on the camera to view live feed

3. **Test detection:**
   - Walk in front of the camera
   - You should see detection boxes appear
   - Events will be recorded automatically

---

## ▶️ Starting & Stopping Frigate

### Starting Frigate

```bash
cd ~/frigate

# Start Frigate
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f frigate
```

### Stopping Frigate

```bash
cd ~/frigate

# Stop Frigate (graceful shutdown)
docker compose stop

# Or stop and remove containers
docker compose down

# Check it's stopped
docker compose ps
# Should show no running containers
```

### Restarting Frigate

```bash
cd ~/frigate

# Restart Frigate
docker compose restart

# Or stop and start
docker compose stop
docker compose up -d
```

### Viewing Status

```bash
# Check if running
docker compose ps

# Check container health
docker ps | grep frigate

# View resource usage
docker stats frigate

# View logs
docker compose logs frigate

# Follow logs in real-time
docker compose logs -f frigate
```

---

## 🖥️ Web Interface Overview

### Main Dashboard

When you access `https://localhost:8971`, you'll see:

1. **Camera Grid** - All your cameras in a grid view
2. **Live View** - Click a camera to see live feed
3. **Events** - Recorded events with detections
4. **Timeline** - Visual timeline of recordings
5. **Settings** - Configuration options

### Key Features

**Live View:**
- Click any camera to view live feed
- See detection boxes in real-time
- Adjust camera settings

**Events:**
- View all recorded events
- Filter by object type (person, car, etc.)
- Download clips and snapshots
- Delete unwanted events

**Timeline:**
- See when recordings occurred
- Navigate through time
- Playback recordings

**Settings:**
- Configure cameras
- Manage users
- Adjust detection settings
- View system information

---

## ⚙️ Basic Configuration

### Global Settings

Edit `config/config.yml` to configure global options:

```yaml
# Database location
database:
  path: /db/frigate.db

# Model configuration (AI detection)
model:
  model_type: yolo-generic
  width: 640
  height: 640
  path: "/config/yolov8n_opset21.onnx"

# MQTT (optional - for Home Assistant integration)
mqtt:
  enabled: false
  # host: broker.emqx.io
  # port: 1883

# Object detection settings
detect:
  enabled: true
  width: 1920
  height: 1080
  fps: 5

# Recording settings
record:
  enabled: true
  retain:
    days: 7

# Snapshots
snapshots:
  enabled: true
  retain:
    days: 30
```

### Camera-Specific Settings

Each camera can have different settings:

```yaml
cameras:
  my_camera:
    enabled: true
    
    # Detection settings
    detect:
      width: 1920
      height: 1080
      fps: 5
      max_disappeared: 25
    
    # Recording
    record:
      enabled: true
      retain:
        days: 7
    
    # Snapshots
    snapshots:
      enabled: true
      retain:
        days: 30
    
    # Motion detection zones
    motion:
      mask:
        - 0,0,100,100  # Ignore motion in this area
    
    # Object filters (only detect specific objects)
    objects:
      track:
        - person
        - car
        - dog
```

---

## 🔧 Troubleshooting

### Problem: Can't Access Web Interface

**Symptoms:** Browser shows "Connection refused" or "This site can't be reached"

**Solutions:**

1. **Check if container is running:**
```bash
docker compose ps
# Should show "Up" status
```

2. **Check port mappings:**
```bash
docker port frigate
# Should show: 8971/tcp -> 0.0.0.0:8971
```

3. **Check firewall:**
```bash
# Ubuntu/Debian
sudo ufw status
sudo ufw allow 8971

# Or check iptables
sudo iptables -L -n | grep 8971
```

4. **Try HTTP instead of HTTPS:**
```
http://localhost:5000
```

5. **Check logs for errors:**
```bash
docker logs frigate 2>&1 | tail -50
```

### Problem: Camera Not Showing

**Symptoms:** Camera added but not visible in web interface

**Solutions:**

1. **Check camera RTSP URL is correct:**
```bash
# Test RTSP URL with ffmpeg
ffmpeg -i rtsp://username:password@ip:port/stream -t 5 test.mp4
```

2. **Check logs for connection errors:**
```bash
docker logs frigate 2>&1 | grep -i "camera_name\|ffmpeg\|error"
```

3. **Verify camera credentials:**
   - Username and password are correct
   - Camera allows RTSP connections
   - Camera IP address is reachable

4. **Check camera is enabled:**
```yaml
cameras:
  my_camera:
    enabled: true  # Must be true
```

5. **Restart Frigate:**
```bash
docker compose restart frigate
```

### Problem: No Detections

**Symptoms:** Camera shows but no objects are detected

**Solutions:**

1. **Check detection is enabled:**
```yaml
detect:
  enabled: true
```

2. **Verify model file exists:**
```bash
docker exec frigate ls -la /config/yolov8n_opset21.onnx
```

3. **Check detection settings:**
```yaml
detect:
  width: 1920   # Match camera resolution
  height: 1080
  fps: 5        # Higher = more CPU, better detection
```

4. **Check object filters:**
```yaml
objects:
  track:
    - person  # Make sure objects you want are listed
    - car
```

5. **Increase detection sensitivity:**
```yaml
detect:
  fps: 10  # Increase from 5 to 10
```

### Problem: High CPU Usage

**Symptoms:** System becomes slow, high CPU usage

**Solutions:**

1. **Reduce detection FPS:**
```yaml
detect:
  fps: 2  # Lower = less CPU
```

2. **Reduce detection resolution:**
```yaml
detect:
  width: 640   # Lower resolution
  height: 480
```

3. **Disable recording for some cameras:**
```yaml
record:
  enabled: false
```

4. **Limit number of cameras:**
   - Start with 1-2 cameras
   - Add more gradually

### Problem: Out of Disk Space

**Symptoms:** Recordings stop, errors about disk space

**Solutions:**

1. **Reduce retention days:**
```yaml
record:
  retain:
    days: 3  # Keep fewer days

snapshots:
  retain:
    days: 7  # Keep fewer snapshots
```

2. **Check disk usage:**
```bash
df -h
du -sh ~/frigate/input/*
```

3. **Clean up old recordings:**
```bash
# Manually delete old files
find ~/frigate/input -type f -mtime +7 -delete
```

### Problem: Container Keeps Restarting

**Symptoms:** Container status shows "Restarting"

**Solutions:**

1. **Check logs for errors:**
```bash
docker logs frigate 2>&1 | tail -100
```

2. **Check config file syntax:**
```bash
docker compose config
# Should show no errors
```

3. **Check disk space:**
```bash
df -h
```

4. **Check memory:**
```bash
free -h
```

5. **Increase shared memory:**
```yaml
shm_size: '1gb'  # Increase from 512mb
```

---

## 📝 Common Tasks

### Adding a New Camera

1. Edit `config/config.yml`
2. Add camera section under `cameras:`
3. Save file
4. Restart: `docker compose restart frigate`

### Disabling a Camera

```yaml
cameras:
  my_camera:
    enabled: false  # Change to false
```

Then restart: `docker compose restart frigate`

### Changing Recording Retention

```yaml
record:
  retain:
    days: 14  # Change number of days
```

### Updating Frigate

```bash
# Pull new image
docker pull kumar2421/frigate-custom:v1.1.3

# Stop current container
docker compose down

# Start with new image
docker compose up -d
```

### Backing Up Configuration

```bash
# Backup config file
cp ~/frigate/config/config.yml ~/frigate/config/config.yml.backup

# Backup entire config directory
tar -czf frigate-config-backup.tar.gz ~/frigate/config/
```

### Viewing System Resources

```bash
# CPU and memory usage
docker stats frigate

# Disk usage
df -h
du -sh ~/frigate/*

# Container logs
docker logs frigate --tail 100
```

### Exporting Recordings

```bash
# Recordings are stored in:
~/frigate/input/recordings/

# Copy to another location
cp -r ~/frigate/input/recordings/ /path/to/backup/
```

---

## 🎓 Next Steps

Now that you have Frigate running:

1. **Add more cameras** - Follow the camera configuration steps
2. **Configure zones** - Define areas for detection
3. **Set up notifications** - Get alerts for detections
4. **Integrate with Home Assistant** - Connect to your smart home
5. **Fine-tune detection** - Adjust sensitivity and filters
6. **Set up recording schedules** - Record only during specific times

---

## 📚 Additional Resources

- **Frigate Documentation:** https://docs.frigate.video/
- **Community Support:** https://github.com/blakeblackshear/frigate/discussions
- **Configuration Examples:** https://docs.frigate.video/configuration/

---

## ✅ Quick Reference Commands

```bash
# Start Frigate
cd ~/frigate && docker compose up -d

# Stop Frigate
cd ~/frigate && docker compose stop

# Restart Frigate
cd ~/frigate && docker compose restart

# View logs
docker compose logs -f frigate

# Check status
docker compose ps

# Edit config
nano ~/frigate/config/config.yml

# Access web UI
https://localhost:8971
```

---

**Congratulations!** You now have Frigate running and know how to manage it. Happy monitoring! 🎉

