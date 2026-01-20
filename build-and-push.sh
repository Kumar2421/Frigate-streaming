#!/bin/bash
# Safe build and push script for Frigate custom image

set -e

REPO_ROOT="/mnt/additional-disk/frigate"
cd "$REPO_ROOT"

# Configuration
IMAGE_NAME="kumar2421/frigate-custom"
VERSION="${1:-v1.1.1}"
DOCKERFILE="custom-detector/Dockerfile"

echo "🌟 Frigate Custom Image Build & Push Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Verify prerequisites
echo "📋 Step 1: Verifying prerequisites..."
if ! bash "$REPO_ROOT/verify-build.sh"; then
    echo "❌ Verification failed. Aborting build."
    exit 1
fi
echo ""

# Step 2: Confirm build context
echo "📋 Step 2: Confirming build context..."
CURRENT_DIR=$(pwd)
if [ "$CURRENT_DIR" != "$REPO_ROOT" ]; then
    echo "❌ ERROR: Must run from repo root!"
    echo "   Current: $CURRENT_DIR"
    echo "   Expected: $REPO_ROOT"
    exit 1
fi
echo "✓ Build context: $CURRENT_DIR"
echo ""

# Step 3: Build the image
echo "📋 Step 3: Building Docker image..."
echo "   Image: ${IMAGE_NAME}:${VERSION}"
echo "   Dockerfile: $DOCKERFILE"
echo "   Context: . (current directory)"
echo ""
echo "⏳ This may take 10-20 minutes..."
docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${VERSION}" .
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful!"
echo ""

# Step 4: Tag as latest
echo "📋 Step 4: Tagging as latest..."
docker tag "${IMAGE_NAME}:${VERSION}" "${IMAGE_NAME}:latest"
echo "✅ Tagged ${IMAGE_NAME}:latest"
echo ""

# Step 5: Verify image exists
echo "📋 Step 5: Verifying image..."
if docker images | grep -q "${IMAGE_NAME}.*${VERSION}"; then
    echo "✅ Image verified:"
    docker images | grep "${IMAGE_NAME}" | head -2
else
    echo "❌ Image not found after build!"
    exit 1
fi
echo ""

# Step 6: Ask about pushing
echo "📋 Step 6: Push to Docker Hub?"
read -p "   Push ${IMAGE_NAME}:${VERSION} and :latest to Docker Hub? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⏳ Logging in to Docker Hub..."
    docker login
    
    echo "⏳ Pushing ${IMAGE_NAME}:${VERSION}..."
    docker push "${IMAGE_NAME}:${VERSION}"
    
    echo "⏳ Pushing ${IMAGE_NAME}:latest..."
    docker push "${IMAGE_NAME}:latest"
    
    echo "✅ Push complete!"
    echo ""
    echo "📋 Next steps on client machine:"
    echo "   1. Update docker-compose.yml: image: ${IMAGE_NAME}:${VERSION}"
    echo "   2. Run: docker pull ${IMAGE_NAME}:${VERSION}"
    echo "   3. Run: docker compose up -d"
else
    echo "⏭️  Skipping push. Image is ready locally."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Build process complete!"
echo ""

