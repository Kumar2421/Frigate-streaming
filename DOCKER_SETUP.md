# Docker Setup for Frigate on Windows

This guide will help you run Frigate using Docker on Windows, which eliminates all OS-specific compatibility issues.

## 🌟 Benefits of Using Docker

✅ **No Windows Compatibility Issues**: Runs in Linux container, uses native IPC sockets  
✅ **No Port Conflicts**: Isolated networking, no port binding issues  
✅ **Proper File Permissions**: Linux filesystem permissions work correctly  
✅ **Easy Updates**: Just pull new image and restart  
✅ **Isolated Environment**: Doesn't affect your Windows system  
✅ **Production Ready**: Same environment as production deployments  

## 📋 Prerequisites

1. **Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and ensure it's running
   - Enable WSL 2 backend (recommended)

2. **Directory Structure**
   ```
   D:\projects\frigate\
   ├── docker-compose.yml
   ├── config/
   │   └── config.yml
   └── input/
       ├── recordings/
       ├── clips/
       └── exports/
   ```

## 🚀 Quick Start

### 1. Ensure Docker is Running

Open Docker Desktop and verify it's running (green icon in system tray).

### 2. Start Frigate

From the project root directory (`D:\projects\frigate`):

```powershell
docker compose up -d
```

### 3. View Logs

```powershell
docker logs -f frigate
```

### 4. Access Web Interface

Open your browser and go to:
- **HTTPS (Recommended)**: https://localhost:8971
- **HTTP (Internal)**: http://localhost:5000

The admin username and password will be shown in the logs on first startup.

### 5. Stop Frigate

```powershell
docker compose down
```

## 🔧 Configuration

### Using Your Existing Config

Your existing `config/config.yml` will be automatically mounted into the container. The Docker setup will:
- Use Linux paths (`/config`, `/media/frigate`)
- Use IPC sockets (no port conflicts)
- Use proper filesystem permissions
- Use tmpfs for cache (no Windows path issues)

### Updating Config

1. Edit `config/config.yml` on your Windows machine
2. Restart the container:
   ```powershell
   docker compose restart
   ```

## 📁 Volume Mounts

The `docker-compose.yml` mounts:
- `./config` → `/config` (read-only, contains your config.yml)
- `./input` → `/media/frigate` (read-write, for recordings/clips)

## 🔄 Updating Frigate

To update to the latest version:

```powershell
docker compose pull
docker compose up -d
```

## 🐛 Troubleshooting

### Port Already in Use

If you get port conflicts, you can change the ports in `docker-compose.yml`:

```yaml
ports:
  - "8972:8971"  # Change external port
```

### Permission Issues

Docker handles all permissions automatically. If you see permission errors:
1. Ensure Docker Desktop is running with proper permissions
2. Check that volumes are mounted correctly

### View Logs

```powershell
# Follow logs
docker logs -f frigate

# Last 100 lines
docker logs --tail 100 frigate
```

### Restart Container

```powershell
docker compose restart
```

### Remove and Recreate

```powershell
docker compose down
docker compose up -d
```

## 🎯 Advantages Over Native Windows

| Feature | Native Windows | Docker |
|---------|---------------|--------|
| IPC Sockets | ❌ TCP workaround | ✅ Native IPC |
| Port Conflicts | ❌ Common | ✅ Isolated |
| File Permissions | ❌ Windows-specific | ✅ Linux standard |
| Path Issues | ❌ Windows paths | ✅ Linux paths |
| Process Management | ❌ Windows-specific | ✅ Standard Linux |
| Updates | ❌ Manual | ✅ `docker compose pull` |
| Isolation | ❌ System-wide | ✅ Containerized |

## 📝 Development Mode

If you want to test local code changes, you can build from source:

1. Uncomment the build section in `docker-compose.yml`:
   ```yaml
   build:
     context: .
     dockerfile: docker/main/Dockerfile
   ```

2. Comment out the image line:
   ```yaml
   # image: ghcr.io/blakeblackshear/frigate:stable
   ```

3. Build and run:
   ```powershell
   docker compose build
   docker compose up -d
   ```
# View logs
docker logs frigate

# Follow logs in real-time
docker logs -f frigate

## 🔐 Security Notes

- Port 5000 is unauthenticated - only expose if needed
- Port 8971 uses HTTPS with authentication
- Change `FRIGATE_RTSP_PASSWORD` in docker-compose.yml
- Consider using Docker networks for additional isolation

## 📚 Additional Resources

- [Frigate Docker Documentation](https://docs.frigate.video/installation/docker)
- [Docker Desktop Documentation](https://docs.docker.com/desktop/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

**Happy Dockerizing! 🐳**

