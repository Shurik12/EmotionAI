#!/bin/bash

# install_deps.sh - Install dependencies for EmotionAI project
# Usage: sudo ./install_deps.sh

set -e

echo "=========================================="
echo "Installing dependencies for EmotionAI project"
echo "=========================================="

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install core dependencies
echo "Installing core dependencies..."
sudo apt-get install -y \
    libopencv-dev \
    libyaml-cpp-dev \
    libspdlog-dev \
    libfmt-dev \
    nlohmann-json3-dev \
    libhiredis-dev \
    build-essential \
    pkg-config

echo "=========================================="
echo "Verifying installations"
echo "=========================================="

# Verify installations
echo "Verifying OpenCV:"
pkg-config --modversion opencv4 || pkg-config --modversion opencv

echo "Verifying yaml-cpp:"
pkg-config --modversion yaml-cpp

echo "Verifying spdlog:"
pkg-config --modversion spdlog

echo "Verifying fmt:"
pkg-config --modversion fmt

echo "Verifying nlohmann-json:"
pkg-config --modversion nlohmann_json

echo "=========================================="
echo "Installation complete!"
echo "=========================================="

echo "Dependencies installed:"
echo "✓ OpenCV"
echo "✓ yaml-cpp"
echo "✓ spdlog"
echo "✓ fmt"
echo "✓ nlohmann-json"
echo "✓ PostgreSQL client"
echo "✓ hiredis"
echo "✓ httplib (header-only)"
echo "✓ redis-plus-plus (headers)"
