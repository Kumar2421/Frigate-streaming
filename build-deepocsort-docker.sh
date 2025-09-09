#!/bin/bash

# Build custom Frigate Docker image with DeepOCSORT support

echo "🌟 Building custom Frigate Docker image with DeepOCSORT support..."

# Build the custom image
echo "📦 Building Docker image..."
docker build -f Dockerfile.deepocsort -t frigate-deepocsort:latest .

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✅ Docker image built successfully!"

# Create config directory if it doesn't exist
if [ ! -d "config" ]; then
    echo "📁 Creating config directory..."
    mkdir -p config
fi

# Copy example config if config.yml doesn't exist
if [ ! -f "config/config.yml" ]; then
    echo "📋 Copying example configuration..."
    cp "config/config.yml.example" "config/config.yml"
fi

echo ""
echo "🚀 Setup complete! You can now run Frigate with DeepOCSORT using:"
echo ""
echo "   docker-compose -f docker-compose.deepocsort.yml up -d"
echo ""
echo "Or run directly with:"
echo "   docker run --rm --publish=5000:5000 --volume=\"\$(pwd)/config:/config\" frigate-deepocsort:latest"
echo ""
