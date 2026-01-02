# 🐳 Docker Migration Guide for Frigate with DeepOCSORT

This guide explains how to move your Frigate project with DeepOCSORT to another machine using Docker.

## 📋 **Prerequisites**

- Docker and Docker Compose installed on both machines
- Access to both source and destination machines
- Network connectivity between machines (for file transfer)

## 🚀 **Method 1: Complete Project Transfer (Recommended)**

### **Step 1: Prepare the Source Machine**

1. **Stop the current Frigate container:**
   ```bash
   docker-compose -f docker-compose.deepocsort.yml down
   # or
   docker stop frigate
   ```

2. **Create a backup of your project:**
   ```bash
   # Create a tar archive of the entire project
   tar -czf frigate-deepocsort-backup.tar.gz \
     --exclude='node_modules' \
     --exclude='.git' \
     --exclude='*.log' \
     --exclude='input/clips' \
     --exclude='input/recordings' \
     .
   ```

   **Windows PowerShell:**
   ```powershell
   # Create a zip archive
   Compress-Archive -Path ".\*" -DestinationPath "frigate-deepocsort-backup.zip" -Exclude @("node_modules", ".git", "*.log", "input\clips", "input\recordings")
   ```

### **Step 2: Transfer to Destination Machine**

**Option A: Using SCP (Linux/macOS):**
```bash
scp frigate-deepocsort-backup.tar.gz user@destination-machine:/path/to/destination/
```

**Option B: Using USB Drive:**
1. Copy the archive to a USB drive
2. Transfer to the destination machine

**Option C: Using Cloud Storage:**
1. Upload to Google Drive, Dropbox, etc.
2. Download on destination machine

### **Step 3: Setup on Destination Machine**

1. **Extract the project:**
   ```bash
   tar -xzf frigate-deepocsort-backup.tar.gz -C /path/to/destination/
   cd /path/to/destination/frigate
   ```

   **Windows:**
   ```powershell
   Expand-Archive -Path "frigate-deepocsort-backup.zip" -DestinationPath "C:\path\to\destination\"
   cd C:\path\to\destination\frigate
   ```

2. **Build the Docker image:**
   ```bash
   # Linux/macOS
   chmod +x build-deepocsort-docker.sh
   ./build-deepocsort-docker.sh
   
   # Windows
   build-deepocsort-docker.bat
   ```

3. **Start Frigate:**
   ```bash
   docker-compose -f docker-compose.deepocsort.yml up -d
   ```

## 🚀 **Method 2: Docker Image Export/Import**

### **Step 1: Export Docker Image from Source**

```bash
# Build the image first (if not already built)
docker build -f Dockerfile.deepocsort -t frigate-deepocsort:latest .

# Export the image
docker save frigate-deepocsort:latest | gzip > frigate-deepocsort-image.tar.gz
```

### **Step 2: Transfer and Import on Destination**

```bash
# Transfer the image file (using your preferred method)
# Then import on destination machine:
docker load < frigate-deepocsort-image.tar.gz

# Verify the image is loaded
docker images | grep frigate-deepocsort
```

### **Step 3: Setup Project Files**

1. **Copy only the essential files:**
   ```bash
   # Create minimal project structure
   mkdir -p frigate-project/{config,web}
   
   # Copy configuration
   cp config/config.yml frigate-project/config/
   
   # Copy web interface files
   cp -r web/src frigate-project/web/
   cp web/package.json frigate-project/web/
   cp web/tsconfig.json frigate-project/web/
   ```

2. **Copy Docker files:**
   ```bash
   cp Dockerfile.deepocsort frigate-project/
   cp docker-compose.deepocsort.yml frigate-project/
   cp build-deepocsort-docker.* frigate-project/
   ```

## 🚀 **Method 3: Git Repository (Best for Development)**

### **Step 1: Create Git Repository**

```bash
# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << EOF
# Docker
*.log
input/clips/
input/recordings/
config/frigate.db*

# Python
__pycache__/
*.pyc
*.pyo

# Node.js
node_modules/
npm-debug.log*

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Add and commit files
git add .
git commit -m "Initial commit: Frigate with DeepOCSORT"
```

### **Step 2: Push to Remote Repository**

```bash
# Create repository on GitHub/GitLab
# Add remote origin
git remote add origin https://github.com/yourusername/frigate-deepocsort.git
git push -u origin main
```

### **Step 3: Clone on Destination Machine**

```bash
# Clone the repository
git clone https://github.com/yourusername/frigate-deepocsort.git
cd frigate-deepocsort

# Build and run
./build-deepocsort-docker.sh
docker-compose -f docker-compose.deepocsort.yml up -d
```

## ⚙️ **Configuration Updates for New Machine**

### **Update Camera URLs**

Edit `config/config.yml` and update camera URLs for the new network:

```yaml
cameras:
  your_camera:
    ffmpeg:
      inputs:
        - path: rtsp://NEW_IP_ADDRESS:554/stream  # Update IP
          roles:
            - detect
            - record
```

### **Update Network Settings**

```yaml
# Update MQTT broker if needed
mqtt:
  host: NEW_MQTT_BROKER_IP  # Update if MQTT broker changed
  port: 1883
```

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **Port Conflicts:**
   ```bash
   # Check if port 5000 is in use
   netstat -tulpn | grep :5000
   
   # Use different port
   docker run --rm --publish=5001:5000 --volume="$(pwd)/config:/config" frigate-deepocsort:latest
   ```

2. **Permission Issues:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER config/
   chmod -R 755 config/
   ```

3. **Docker Image Not Found:**
   ```bash
   # Rebuild the image
   docker build -f Dockerfile.deepocsort -t frigate-deepocsort:latest .
   ```

### **Verification Steps:**

1. **Check container status:**
   ```bash
   docker ps | grep frigate
   ```

2. **Check logs:**
   ```bash
   docker logs frigate
   ```

3. **Test web interface:**
   ```bash
   curl http://localhost:5000
   ```

## 📊 **File Size Estimates**

| Component | Size | Notes |
|-----------|------|-------|
| Docker Image | ~2-3 GB | Includes all dependencies |
| Config Files | ~1-10 MB | Depends on configuration complexity |
| Web Interface | ~50-100 MB | React build files |
| Project Archive | ~100-200 MB | Complete project (excluding media) |

## 🎯 **Quick Migration Checklist**

- [ ] Stop Frigate on source machine
- [ ] Create project backup
- [ ] Transfer files to destination
- [ ] Install Docker on destination
- [ ] Build custom Docker image
- [ ] Update configuration for new network
- [ ] Start Frigate on destination
- [ ] Verify web interface access
- [ ] Test camera feeds
- [ ] Verify DeepOCSORT functionality

## 🚀 **Automated Migration Script**

```bash
#!/bin/bash
# migrate-frigate.sh

SOURCE_MACHINE="user@source-ip"
DEST_MACHINE="user@dest-ip"
PROJECT_PATH="/opt/frigate"

echo "🚀 Starting Frigate migration..."

# Create backup on source
ssh $SOURCE_MACHINE "cd $PROJECT_PATH && tar -czf /tmp/frigate-backup.tar.gz --exclude='input/clips' --exclude='input/recordings' ."

# Transfer backup
scp $SOURCE_MACHINE:/tmp/frigate-backup.tar.gz /tmp/

# Setup on destination
ssh $DEST_MACHINE "mkdir -p $PROJECT_PATH && cd $PROJECT_PATH && tar -xzf /tmp/frigate-backup.tar.gz"

# Build and start
ssh $DEST_MACHINE "cd $PROJECT_PATH && ./build-deepocsort-docker.sh && docker-compose -f docker-compose.deepocsort.yml up -d"

echo "✅ Migration complete!"
```

---

**Happy Migrating! 🎉**
