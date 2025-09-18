#!/bin/bash

# install_deps.sh - Install dependencies for EmotionAI project
# Usage: sudo ./install_deps.sh

set -e

echo "=========================================="
echo "Installing dependencies for EmotionAI project"
echo "=========================================="

# Install submodules
echo "Installing git submodules..."
git submodule update --init --recursive
git apply emotiefflib.patch

# Install onnx
echo "Installing onnx..."
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz -P contrib/
tar zxvf contrib/onnxruntime-linux-x64-1.21.0.tgz -C contrib/
mv contrib/onnxruntime-linux-x64-1.21.0 contrib/onnxruntime
rm contrib/onnxruntime-linux-x64-1.21.0.tgz
mkdir contrib/onnxruntime/lib64
mv contrib/onnxruntime/lib/*.so* contrib/onnxruntime/lib64/
cp contrib/onnxruntime/lib/cmake/onnxruntime/onnxruntimeConfig.cmake contrib/onnxruntime/lib/cmake/onnxruntime/ONNXRuntimeConfig.cmake

# Install torch
echo "Installing libtorch..."
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip -d contrib/
rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

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
