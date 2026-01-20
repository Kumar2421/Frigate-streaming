# 🔧 Deployment Fix: Config Directory Issue

## Problem
The container was failing with:
```
FileNotFoundError: [Errno 2] No such file or directory: '/config/config.yaml'
```

This happened because:
1. The `/config` directory wasn't guaranteed to exist
2. Even if mounted, the directory might not be writable
3. Frigate couldn't create a default config file

## Solution
Created an enhanced entrypoint script (`custom-detector/entrypoint.sh`) that:

1. **Ensures `/config` directory exists** - Creates it if missing
2. **Ensures `/db` directory exists** - Creates it for database storage
3. **Ensures `/media/frigate` directory exists** - Creates it for recordings/clips
4. **Copies default config** - If no config exists, copies from `/opt/frigate/config_defaults/config.yml`
5. **Copies model file** - Ensures `yolov8n_opset21.onnx` is available
6. **Sets proper permissions** - Makes all directories writable
7. **Then starts Frigate** - Calls the original `/init` entrypoint

## Changes Made

### 1. Enhanced Entrypoint Script
- **File**: `custom-detector/entrypoint.sh`
- **Purpose**: Initialize config directory before Frigate starts
- **Features**:
  - Auto-creates `/config` if missing
  - Copies default config from bundled defaults
  - Handles both `config.yml` and `config.yaml`
  - Copies model files if needed

### 2. Updated Dockerfile
- **Change**: Uses custom entrypoint instead of direct `/init`
- **Line 131**: `ENTRYPOINT ["/opt/frigate/entrypoint.sh"]`
- **Benefit**: Config initialization happens automatically

## Rebuild Instructions

### Step 1: Verify Files
```bash
cd /mnt/additional-disk/frigate
./verify-build.sh
```

### Step 2: Rebuild Image
```bash
docker build -f custom-detector/Dockerfile \
  -t kumar2421/frigate-custom:v1.1.3 .
```

### Step 3: Test Locally
```bash
# Test that entrypoint works
docker run --rm kumar2421/frigate-custom:v1.1.3 \
  ls -la /config

# Should show config.yml was created
```

### Step 4: Tag and Push
```bash
docker tag kumar2421/frigate-custom:v1.1.3 kumar2421/frigate-custom:latest
docker login
docker push kumar2421/frigate-custom:v1.1.3
docker push kumar2421/frigate-custom:latest
```

## Deployment on Client

### Option 1: With docker-compose (Recommended)
```bash
# Update docker-compose.yml
# image: kumar2421/frigate-custom:v1.1.3

docker compose down
docker compose up -d
docker compose logs -f frigate
```

### Option 2: Standalone Run
```bash
# Pull new image
docker pull kumar2421/frigate-custom:v1.1.3

# Run with volume mount (config will be auto-created if missing)
docker run -d \
  --name frigate \
  --restart unless-stopped \
  -v ~/frigate-config:/config:rw \
  -v ~/frigate-media:/media/frigate:rw \
  --shm-size=512mb \
  -p 5000:5000 \
  -p 5001:5001 \
  -p 8554:8554 \
  -p 8555:8555/tcp \
  -p 8555:8555/udp \
  -e TZ=Asia/Kolkata \
  kumar2421/frigate-custom:v1.1.3
```

## What the Entrypoint Does

1. **Checks for `/config` directory** - Creates if missing
2. **Checks for `/db` directory** - Creates if missing (for database)
3. **Checks for `/media/frigate` directory** - Creates if missing (for recordings)
4. **Checks for config file** - Copies default if none exists
5. **Copies model file** - Ensures ONNX model is available
6. **Initializes DeepSORT** - If config files exist
7. **Starts Frigate** - Calls original entrypoint

## Benefits

✅ **No manual config setup needed** - Works out of the box  
✅ **Handles missing volumes** - Creates config if not mounted  
✅ **Backward compatible** - Works with existing setups  
✅ **Better error messages** - Clear logging of what's happening  

## Verification

After deployment, check logs:
```bash
docker logs frigate 2>&1 | grep -E "Starting|config|DeepSORT"
```

You should see:
```
🌟 Starting Enhanced Frigate with DeepSORT tracking...
⚠️  /config directory doesn't exist, creating it...
⚠️  /db directory doesn't exist, creating it...
⚠️  /media/frigate directory doesn't exist, creating it...
📝 No config file found, initializing from defaults...
📋 Copying default config from image...
✅ Default config copied to /config/config.yml
✅ DeepSORT available
🚀 Starting Frigate...
```

No more `FileNotFoundError` or database errors! 🎉

## Important Note

⚠️ **For production use, always use docker-compose with proper volume mounts:**
- `./config:/config:rw` - For config files
- `frigate_db:/db` - For database (named volume)
- `./input:/media/frigate:rw` - For recordings/clips

The entrypoint creates directories as a fallback, but volumes ensure data persistence across container restarts.

