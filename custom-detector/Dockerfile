# Enhanced Frigate with DeepSORT Tracking
FROM ghcr.io/blakeblackshear/frigate:stable

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies for DeepSORT and Node.js for web rebuild
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages with --break-system-packages flag
RUN pip3 install --upgrade pip setuptools wheel --break-system-packages

# Fix NumPy compatibility issue - downgrade if needed
RUN pip3 install --force-reinstall "numpy>=1.24.0,<2.0.0" --break-system-packages

# Install PyTorch (CPU version for compatibility)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Install DeepSORT and tracking dependencies
RUN pip3 install \
    deep-sort-realtime \
    filterpy \
    lap \
    cython \
    cython-bbox \
    --break-system-packages

# Install additional computer vision packages
RUN pip3 install \
    opencv-python \
    opencv-contrib-python \
    "numpy>=1.24.0,<2.0.0" \
    scipy \
    pillow \
    tqdm \
    pyyaml \
    requests \
    psutil \
    --break-system-packages

# Create custom tracker directory
RUN mkdir -p /opt/frigate/custom_trackers

# Copy custom DeepSORT tracker
COPY deepsort_tracker.py /opt/frigate/custom_trackers/
COPY requirements.txt /opt/frigate/custom_trackers/

# Install custom tracker requirements
RUN cd /opt/frigate/custom_trackers && pip3 install -r requirements.txt --break-system-packages

# Create patch script to integrate DeepSORT tracker
COPY patch_frigate.py /opt/frigate/custom_trackers/
COPY integrate_deepsort.py /opt/frigate/custom_trackers/
RUN chmod +x /opt/frigate/custom_trackers/patch_frigate.py
RUN chmod +x /opt/frigate/custom_trackers/integrate_deepsort.py

# Apply the patch to integrate DeepSORT tracker
RUN cd /opt/frigate/custom_trackers && python3 patch_frigate.py

# Copy modified birdseye.py to use logo.png instead of birdseye.png
COPY birdseye.py /opt/frigate/frigate/output/birdseye.py

# Copy logo.png and replace birdseye.png with it
COPY logo.png /opt/frigate/frigate/images/logo.png
RUN cp /opt/frigate/frigate/images/logo.png /opt/frigate/frigate/images/birdseye.png

# Copy web source files for rebuilding
COPY web-source /tmp/web-source

# Copy modified Logo component to web source
COPY Logo.tsx /tmp/web-source/src/components/Logo.tsx
COPY logo.png /tmp/web-source/public/images/logo.png

# Rebuild web app with modified Logo component
WORKDIR /tmp/web-source
RUN npm install && npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ 2>/dev/null || true \
    && rm -rf dist/BASE_PATH

# Copy rebuilt web files to final location
RUN cp -r dist/* /opt/frigate/web/

# Set working directory
WORKDIR /opt/frigate

# Expose ports (same as original Frigate)
EXPOSE 5000 8554 8555 8556

# Use original Frigate entrypoint
ENTRYPOINT ["/init"]