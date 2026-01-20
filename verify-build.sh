#!/bin/bash
# Verification script for Docker build - ensures all required files exist

set -e

REPO_ROOT="/mnt/additional-disk/frigate"
cd "$REPO_ROOT"

echo "🌟 Verifying Docker build prerequisites..."
echo ""

ERRORS=0

# Function to check file/directory
check_file() {
    local file="$1"
    local desc="${2:-$file}"
    if [ -e "$file" ]; then
        echo "✓ $desc"
        return 0
    else
        echo "✗ MISSING: $desc"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

# Check custom-detector files
echo "📁 Checking custom-detector files..."
check_file "custom-detector/deepsort_tracker.py" "custom-detector/deepsort_tracker.py"
check_file "custom-detector/requirements.txt" "custom-detector/requirements.txt"
check_file "custom-detector/patch_frigate.py" "custom-detector/patch_frigate.py"
check_file "custom-detector/integrate_deepsort.py" "custom-detector/integrate_deepsort.py"
check_file "custom-detector/birdseye.py" "custom-detector/birdseye.py"
check_file "custom-detector/logo.png" "custom-detector/logo.png"
check_file "custom-detector/Logo.tsx" "custom-detector/Logo.tsx"
check_file "custom-detector/entrypoint.sh" "custom-detector/entrypoint.sh"
check_file "custom-detector/Dockerfile" "custom-detector/Dockerfile"
check_file "custom-detector/web-source" "custom-detector/web-source (directory)"

# Check frigate files
echo ""
echo "📁 Checking frigate files..."
check_file "frigate/const.py" "frigate/const.py"

# Check config files
echo ""
echo "📁 Checking config files..."
check_file "config/config.yml" "config/config.yml"
# deepsort_config.yml is optional (DeepSORT integration is disabled in entrypoint)
check_file "config/yolov8n_opset21.onnx" "config/yolov8n_opset21.onnx"
check_file "config/yolov8n.onnx" "config/yolov8n.onnx"

# Check web-source structure
echo ""
echo "📁 Checking web-source structure..."
check_file "custom-detector/web-source/package.json" "custom-detector/web-source/package.json"
check_file "custom-detector/web-source/src/components" "custom-detector/web-source/src/components (directory)"
check_file "custom-detector/web-source/public/images" "custom-detector/web-source/public/images (directory)"

# Verify we're in the right directory
echo ""
echo "📁 Verifying build context..."
CURRENT_DIR=$(pwd)
if [ "$CURRENT_DIR" != "$REPO_ROOT" ]; then
    echo "✗ ERROR: Not in repo root! Current: $CURRENT_DIR, Expected: $REPO_ROOT"
    ERRORS=$((ERRORS + 1))
else
    echo "✓ Build context: $CURRENT_DIR"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $ERRORS -eq 0 ]; then
    echo "✅ All files verified! Ready to build."
    echo ""
    echo "To build, run from repo root:"
    echo "  docker build -f custom-detector/Dockerfile -t kumar2421/frigate-custom:v1.1.1 ."
    exit 0
else
    echo "❌ Found $ERRORS missing file(s). Please fix before building."
    exit 1
fi

