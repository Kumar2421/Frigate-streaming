# 🚀 Quick Start Guide

## Start Services
```bash
cd /mnt/additional-disk/frigate
docker compose up -d
```

## View Logs
```bash
# All services
docker compose logs -f

# Frigate only
docker compose logs -f frigate

# Last 100 lines
docker compose logs --tail=100 frigate
```

## Stop Services
```bash
docker compose down
```

## Restart Services
```bash
docker compose restart
```

## Rebuild After Code Changes
```bash
docker compose build frigate
docker compose up -d frigate
```

## Check Status
```bash
docker compose ps
```

## Access Web UI
Open browser: `http://localhost:5000`

## Check Disk Space
```bash
df -h
du -sh /mnt/additional-disk/frigate/*
```

## Clean Docker (Free Space)
```bash
# Remove unused images/containers
docker system prune -a

# Remove build cache
docker builder prune -a
```

## Check DeepSORT Configuration
```bash
docker exec frigate env | grep DEEPSORT
```
