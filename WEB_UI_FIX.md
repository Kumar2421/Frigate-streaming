# 🌐 Web UI Access Fix

## Problem
Ports are not mapped to the host, so you can't access the web UI from your browser.

## Solution

### Option 1: Use Docker Compose (Recommended)

```bash
cd /mnt/additional-disk/frigate

# Stop current container if running
docker stop $(docker ps | grep frigate-custom | awk '{print $1}')

# Update docker-compose.yml to use your image
# Change line 7: image: kumar2421/frigate-custom:v1.1.3

# Start with docker-compose (this maps ports correctly)
docker compose up -d

# Check it's running
docker compose ps

# Access web UI at:
# https://localhost:8971 (or http://localhost:8971 if TLS disabled)
```

### Option 2: Run with Port Mappings

If you prefer `docker run`, use these port mappings:

```bash
# Stop current container
docker stop $(docker ps | grep frigate-custom | awk '{print $1}')

# Run with proper port mappings
docker run -d \
  --name frigate \
  --restart unless-stopped \
  -p 8971:8971 \
  -p 5000:5000 \
  -p 5001:5001 \
  -p 8554:8554 \
  -p 8555:8555/tcp \
  -p 8555:8555/udp \
  -v $(pwd)/config:/config:rw \
  -v frigate_db:/db \
  -v $(pwd)/input:/media/frigate:rw \
  --shm-size=512mb \
  -e TZ=Asia/Kolkata \
  -e FRIGATE_RTSP_PASSWORD=password \
  kumar2421/frigate-custom:v1.1.3
```

## Accessing the Web UI

### Port 8971 (Main Web UI - HTTPS)
```bash
# In browser:
https://localhost:8971
# or
https://YOUR_SERVER_IP:8971

# You may see a certificate warning (self-signed cert) - accept it
```

### Port 5000 (Internal API - HTTP, unauthenticated)
```bash
# In browser:
http://localhost:5000
# Note: This is for internal use, may not show full UI
```

### Port 5001 (Internal API - HTTP, authenticated)
```bash
# In browser:
http://localhost:5001
# Note: Requires authentication
```

## Verify Ports Are Mapped

After starting, verify ports are accessible:

```bash
# Check port mappings
docker port frigate
# Should show:
# 8971/tcp -> 0.0.0.0:8971
# 5000/tcp -> 0.0.0.0:5000
# etc.

# Test from host
curl -k https://localhost:8971 | head -20
# Should return HTML, not "Connection refused"
```

## Troubleshooting

### If you see "Connection refused":
- Ports are not mapped - use docker-compose or add `-p` flags

### If you see "400 Bad Request" or certificate warning:
- Port 8971 uses HTTPS - use `https://` not `http://`
- Accept the self-signed certificate warning

### If you see blank page:
- Wait a few seconds for Frigate to fully start
- Check logs: `docker logs frigate | tail -50`

### If ports are mapped but still can't access:
- Check firewall: `sudo ufw status` or `sudo iptables -L`
- Try accessing from server itself: `curl -k https://localhost:8971`
- Check if another service is using the port: `netstat -tlnp | grep 8971`

