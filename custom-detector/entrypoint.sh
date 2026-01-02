#!/bin/bash
# Enhanced Frigate entrypoint with DeepSORT integration

set -e

echo "Starting Enhanced Frigate with DeepSORT tracking..."

# Check if DeepSORT dependencies are available
python3 -c "import deep_sort_realtime" 2>/dev/null && echo "DeepSORT available" || echo "DeepSORT not available, using fallback"

# Set environment variables for DeepSORT
export DEEPSORT_ENABLED=${DEEPSORT_ENABLED:-true}
export DEEPSORT_MAX_AGE=${DEEPSORT_MAX_AGE:-30}
export DEEPSORT_N_INIT=${DEEPSORT_N_INIT:-3}
export DEEPSORT_MAX_IOU_DISTANCE=${DEEPSORT_MAX_IOU_DISTANCE:-0.7}
export DEEPSORT_MAX_COSINE_DISTANCE=${DEEPSORT_MAX_COSINE_DISTANCE:-0.2}
export DEEPSORT_NN_BUDGET=${DEEPSORT_NN_BUDGET:-100}

# Integrate DeepSORT configuration if config files exist
if [ -f "/config/config.yml" ] && [ -f "/config/deepsort_config.yml" ]; then
    echo "Integrating DeepSORT configuration..."
    cd /opt/frigate/custom_trackers
    python3 integrate_deepsort.py
fi

# Start the original Frigate
exec /init "$@"
