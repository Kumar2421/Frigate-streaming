# 🚀 Frigate Custom Image Build Guide

run this -------------------->
docker build -f custom-detector/Dockerfile -t kumar2421/frigate-custom:1.0.1 .

## ⚠️ IMPORTANT: Always Build from Repo Root

**The build context MUST be the repo root (`/mnt/additional-disk/frigate`), NOT the `custom-detector` directory!**

### ❌ WRONG (will fail):
```bash
cd /mnt/additional-disk/frigate/custom-detector
docker build -f Dockerfile -t kumar2421/frigate-custom:v1.1.1 .
```

### ✅ CORRECT:
```bash
cd /mnt/additional-disk/frigate
docker build -f custom-detector/Dockerfile -t kumar2421/frigate-custom:v1.1.1 .
```

---

## 📋 Quick Build Process

### Option 1: Automated Script (Recommended)

```bash
cd /mnt/additional-disk/frigate

# Step 1: Verify all files exist
./verify-build.sh

# Step 2: Build and push (will prompt for push confirmation)
./build-and-push.sh v1.1.1
```

### Option 2: Manual Build

```bash
cd /mnt/additional-disk/frigate

# Step 1: Verify prerequisites
./verify-build.sh

# Step 2: Build the image
docker build -f custom-detector/Dockerfile \
  -t kumar2421/frigate-custom:v1.1.1 .

# Step 3: Tag as latest
docker tag kumar2421/frigate-custom:v1.1.1 kumar2421/frigate-custom:latest

# Step 4: Verify image
docker images | grep frigate-custom

# Step 5: Push to Docker Hub
docker login
docker push kumar2421/frigate-custom:v1.1.1
docker push kumar2421/frigate-custom:latest
```

---

## ✅ Pre-Build Checklist

Before building, ensure:

- [x] You're in `/mnt/additional-disk/frigate` (repo root)
- [x] All files verified by `./verify-build.sh`
- [x] Docker is running (`docker ps`)
- [x] You're logged into Docker Hub (if pushing): `docker login`

---

## 📦 Files Included in Image

The Dockerfile copies these files into the image:

### Custom Trackers:
- `custom-detector/deepsort_tracker.py`
- `custom-detector/requirements.txt`
- `custom-detector/patch_frigate.py`
- `custom-detector/integrate_deepsort.py`

### Modified Frigate Files:
- `custom-detector/birdseye.py` → `/opt/frigate/frigate/output/birdseye.py`
- `frigate/const.py` → `/opt/frigate/frigate/const.py` (ensures `UPDATE_BIRDSEYE_LAYOUT` exists)

### Assets:
- `custom-detector/logo.png` → `/opt/frigate/frigate/images/logo.png`

### Default Configs & Models:
- `config/config.yml` → `/opt/frigate/config_defaults/config.yml`
- `config/deepsort_config.yml` → `/opt/frigate/config_defaults/deepsort_config.yml`
- `config/yolov8n_opset21.onnx` → `/opt/frigate/config_defaults/yolov8n_opset21.onnx`
- `config/yolov8n.onnx` → `/opt/frigate/config_defaults/yolov8n.onnx`

### Web Interface:
- `custom-detector/web-source/` → rebuilt and copied to `/opt/frigate/web/`
- `custom-detector/Logo.tsx` → replaces default logo component

---

## 🔍 Troubleshooting

### Error: "lstat custom-detector: no such file or directory"
**Cause:** Running build from wrong directory  
**Fix:** `cd /mnt/additional-disk/frigate` first, then build

### Error: "cannot import name 'UPDATE_BIRDSEYE_LAYOUT'"
**Cause:** Old image without updated `const.py`  
**Fix:** Rebuild with latest Dockerfile (includes `frigate/const.py` copy)

### Error: "No such file or directory" during COPY
**Cause:** Missing file in repo  
**Fix:** Run `./verify-build.sh` to identify missing files

---

## 📤 Deployment on Client Machine

After pushing to Docker Hub:

```bash
# On client machine
cd /path/to/frigate

# Update docker-compose.yml:
#   image: kumar2421/frigate-custom:v1.1.1

# Pull and start
docker pull kumar2421/frigate-custom:v1.1.1   # or latest docker build -f custom-detector/Dockerfile -t kumar2421/frigate-custom:1.0.1 .
docker compose up -d
docker logs frigate
```

---

## 🎯 Version Tags

- `v1.1.1` - Current version with `const.py` fix
- `latest` - Always points to most recent stable build

Always tag both version and `latest` for easy updates.

