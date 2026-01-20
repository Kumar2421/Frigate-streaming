#!/bin/bash
# Enhanced Frigate entrypoint with config initialization and DeepSORT integration

set -e

echo "🌟 Starting Enhanced Frigate with DeepSORT tracking..."

# Ensure /config directory exists and is writable
if [ ! -d "/config" ]; then
    echo "⚠️  /config directory doesn't exist, creating it..."
    mkdir -p /config
fi

# Ensure /config is writable
chmod 755 /config 2>/dev/null || true

# Ensure /db directory exists and is writable (for database)
if [ ! -d "/db" ]; then
    echo "⚠️  /db directory doesn't exist, creating it..."
    mkdir -p /db
fi

# Ensure /db is writable
chmod 755 /db 2>/dev/null || true

# Ensure /media/frigate directory exists (for recordings/clips)
if [ ! -d "/media/frigate" ]; then
    echo "⚠️  /media/frigate directory doesn't exist, creating it..."
    mkdir -p /media/frigate
fi

# Ensure /media/frigate is writable
chmod 755 /media/frigate 2>/dev/null || true

# Initialize config file if it doesn't exist
CONFIG_FILE="/config/config.yml"
if [ ! -f "$CONFIG_FILE" ] && [ ! -f "/config/config.yaml" ]; then
    echo "📝 No config file found, initializing from defaults..."
    
    # Copy default config if available
    if [ -f "/opt/frigate/config_defaults/config.yml" ]; then
        echo "📋 Copying default config from image..."
        cp /opt/frigate/config_defaults/config.yml "$CONFIG_FILE"
        echo "✅ Default config copied to $CONFIG_FILE"
    else
        echo "⚠️  No default config found in image, Frigate will create one"
    fi
fi

# Copy model file if it doesn't exist
if [ ! -f "/config/yolov8n_opset21.onnx" ] && [ -f "/opt/frigate/config_defaults/yolov8n_opset21.onnx" ]; then
    echo "📋 Copying default model file..."
    cp /opt/frigate/config_defaults/yolov8n_opset21.onnx /config/yolov8n_opset21.onnx
fi

# Check if DeepSORT dependencies are available
if python3 -c "import deep_sort_realtime" 2>/dev/null; then
    echo "✅ DeepSORT available"
else
    echo "⚠️  DeepSORT not available, using fallback tracker"
fi

# Set environment variables for DeepSORT
export DEEPSORT_ENABLED=${DEEPSORT_ENABLED:-true}
export DEEPSORT_MAX_AGE=${DEEPSORT_MAX_AGE:-30}
export DEEPSORT_N_INIT=${DEEPSORT_N_INIT:-3}
export DEEPSORT_MAX_IOU_DISTANCE=${DEEPSORT_MAX_IOU_DISTANCE:-0.7}
export DEEPSORT_MAX_COSINE_DISTANCE=${DEEPSORT_MAX_COSINE_DISTANCE:-0.2}
export DEEPSORT_NN_BUDGET=${DEEPSORT_NN_BUDGET:-100}

# Integrate DeepSORT configuration if config files exist
if [ -f "/config/config.yml" ] && [ -f "/config/deepsort_config.yml" ]; then
    echo "🔧 Integrating DeepSORT configuration..."
    cd /opt/frigate/custom_trackers
    python3 integrate_deepsort.py || echo "⚠️  DeepSORT integration failed, continuing anyway"
fi

# Start the original Frigate
echo "🚀 Starting Frigate..."
exec /init "$@"
