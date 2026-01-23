# Enhanced Frigate with DeepSORT Tracking
FROM ghcr.io/blakeblackshear/frigate:stable

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# NOTE: Build with root context so we can copy config/models:
#   docker build -f custom-detector/Dockerfile -t yourname/frigate-custom:latest .

# Install system dependencies for DeepSORT
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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages with --break-system-packages flag
RUN pip3 install --upgrade pip setuptools wheel --break-system-packages

# Fix NumPy compatibility issue - downgrade if needed
RUN pip3 install --force-reinstall "numpy>=1.24.0,<2.0.0" --break-system-packages

# Install PyTorch (CPU version for compatibility)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# Install DeepOCSORT + torchreid for ReID-based persistent IDs
# PyPI package name is deep-oc-sort (module imports as deepocsort)
RUN pip3 install \
    deep-sort-realtime \
    filterpy \
    lap \
    cython-bbox \
    torchreid \
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

# Install optional audio transcription dependencies
# RUN pip3 install \
#     faster-whisper \
#     librosa \
#     soundfile \
#     sherpa-onnx \
#     --break-system-packages

# Install Ultralytics for YOLOv8/YOLO models (.pt)
RUN pip3 install ultralytics --break-system-packages

# Create default config directory
RUN mkdir -p /opt/frigate/config_defaults

# Copy modified birdseye.py to use logo.png instead of birdseye.png
COPY custom-detector/birdseye.py /opt/frigate/frigate/output/birdseye.py

# Ensure const.py in the image matches our local version (adds UPDATE_BIRDSEYE_LAYOUT, etc.)
COPY frigate/const.py /opt/frigate/frigate/const.py

# Ensure model post-processing matches our local version (customizable YOLO thresholds)
COPY frigate/util/model.py /opt/frigate/frigate/util/model.py

# Ensure object filtering/reduction debug changes are included
COPY frigate/util/object.py /opt/frigate/frigate/util/object.py

# Include additional API modules
COPY frigate/api/crops.py /opt/frigate/frigate/api/crops.py

# Patch upstream FastAPI app to register crops router without overriding the whole file
RUN python3 - <<'PY'
import pathlib

path = pathlib.Path('/opt/frigate/frigate/api/fastapi_app.py')
text = path.read_text(encoding='utf-8')

# 1) Ensure crops is imported in the multi-import block
import_block_old = 'from frigate.api import (\n'
if import_block_old not in text:
    raise SystemExit(f'Expected frigate.api import block not found in {path}. Upstream file changed?')

if '    crops,\n' not in text:
    text = text.replace(import_block_old, import_block_old + '    crops,\n')

# 2) Ensure router is included
needle = 'app.include_router(media.router)'
if needle not in text:
    raise SystemExit(f'Expected include_router(media.router) not found in {path}. Upstream file changed?')

if 'app.include_router(crops.router)' not in text:
    text = text.replace(needle, needle + '\n    app.include_router(crops.router)')

path.write_text(text, encoding='utf-8')
PY

RUN python3 - <<'PY'
import pathlib

path = pathlib.Path('/opt/frigate/frigate/util/builtin.py')
text = path.read_text(encoding='utf-8')

old = 'return path.replace(pw, urllib.parse.quote_plus(pw))'
new = """# If the password is already percent-encoded (common for '@' as '%40'),\n        # encoding again will break authentication (e.g. '%40' -> '%2540').\n        if re.search(r\"%[0-9A-Fa-f]{2}\", pw):\n            return path\n        return path.replace(pw, urllib.parse.quote(pw, safe=\"\"))"""

if old not in text:
    raise SystemExit(f'Expected pattern not found in {path}. Upstream file changed?')

text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
PY

# Copy modified object_processing.py to include secondary face/fire model logic
COPY frigate/track/object_processing.py /opt/frigate/frigate/track/object_processing.py

# Ensure camera live config matches our local version
COPY frigate/config/camera/live.py /opt/frigate/frigate/config/camera/live.py

RUN python3 - <<'PY'
import pathlib

path = pathlib.Path('/opt/frigate/frigate/camera/__init__.py')
text = path.read_text(encoding='utf-8')

# Upstream uses a very small frame_queue (maxsize=2). With heavier tracking (e.g. DeepSORT),
# the processing loop can't drain quickly enough and capture drops all frames.
replacements = [
    ('mp.Queue(maxsize=2)', 'mp.Queue(maxsize=50)'),
    ('mp.Queue(maxsize = 2)', 'mp.Queue(maxsize = 50)'),
    ('Queue(maxsize=2)', 'Queue(maxsize=50)'),
    ('Queue(maxsize = 2)', 'Queue(maxsize = 50)'),
]

updated = text
for old, new in replacements:
    updated = updated.replace(old, new)

if updated == text:
    raise SystemExit(f'Expected Queue(maxsize=2) pattern not found in {path}. Upstream file changed?')

path.write_text(updated, encoding='utf-8')
PY

# Copy DeepOCSORT tracker implementation used by the main Frigate tracker pipeline
COPY frigate/track/deepocsort_tracker.py /opt/frigate/frigate/track/deepocsort_tracker.py

RUN python3 - <<'PY'
import pathlib

path = pathlib.Path('/opt/frigate/frigate/app.py')
text = path.read_text(encoding='utf-8')

# Increase detected_frames_queue size. The default is small and can become a bottleneck when
# tracking/processing is heavier (e.g. DeepSORT), causing camera processes to drop frames.
lines = text.splitlines(True)
out = []

in_block = False
patched = False
for line in lines:
    if (
        not in_block
        and 'self.detected_frames_queue' in line
        and 'Queue' in line
        and 'mp.Queue' in line
    ):
        # Start of the detected_frames_queue assignment (may span multiple lines)
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}self.detected_frames_queue = mp.Queue(maxsize=200)\n")
        in_block = True
        patched = True
        continue

    if in_block:
        # Skip lines until we reach the end of the original mp.Queue(...) call.
        # Upstream ends the block with a standalone ')' line.
        if line.strip() == ')':
            in_block = False
            continue
        continue

    out.append(line)

if not patched:
    raise SystemExit(f'Failed to locate detected_frames_queue in {path}. Upstream file changed?')

path.write_text(''.join(out), encoding='utf-8')
PY

# Override Norfair tracker module to use DeepOCSORTTracker (backed by deep-sort-realtime)
# This avoids overriding upstream frigate/video.py (which may differ between image versions).
COPY custom-detector/norfair_tracker.py /opt/frigate/frigate/track/norfair_tracker.py

# Copy logo.png and replace birdseye.png with it
COPY custom-detector/logo.png /opt/frigate/frigate/images/logo.png
RUN cp /opt/frigate/frigate/images/logo.png /opt/frigate/frigate/images/birdseye.png

# Bundle default configs and models so image works without external volumes
COPY config/config.yml /opt/frigate/config_defaults/config.yml
# deepsort_config.yml removed - DeepSORT integration is disabled in entrypoint
COPY config/yolov8n_opset21.onnx /opt/frigate/config_defaults/yolov8n_opset21.onnx
# Optional legacy model if needed by consumers
COPY config/yolov8n.onnx /opt/frigate/config_defaults/yolov8n.onnx

# Copy enhanced entrypoint script
COPY custom-detector/entrypoint.sh /opt/frigate/entrypoint.sh
RUN chmod +x /opt/frigate/entrypoint.sh && sed -i 's/\r$//' /opt/frigate/entrypoint.sh

# Set working directory
WORKDIR /opt/frigate

# Expose ports (same as original Frigate)
EXPOSE 5000 8554 8555 8556

# Use enhanced entrypoint that ensures config directory exists
ENTRYPOINT ["/opt/frigate/entrypoint.sh"]