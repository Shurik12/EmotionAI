#!/usr/bin/env bash
#
# Install ONNX Runtime into contrib/onnxruntime for either the CPU or GPU build.
#
# Both variants share the same layout after install (headers in include/, shared
# libraries in lib64/), so the C++ build, the loader path and the emotiefflib
# CMake patch are identical regardless of which one is installed. The only
# runtime difference is that the GPU build ships libonnxruntime_providers_cuda.so
# and needs CUDA/cuDNN on the loader path.
#
# Usage:
#   scripts/install_onnxruntime.sh [cpu|gpu] [version] [dest]
#
# Examples:
#   scripts/install_onnxruntime.sh gpu            # GPU build (default version)
#   ONNX_VARIANT=gpu make install_onnx            # same via the Makefile
#
set -euo pipefail

VARIANT="${1:-cpu}"
VERSION="${2:-1.21.0}"
DEST="${3:-contrib/onnxruntime}"

case "${VARIANT}" in
  cpu)
    PKG="onnxruntime-linux-x64-${VERSION}"
    ;;
  gpu)
    PKG="onnxruntime-linux-x64-gpu-${VERSION}"
    ;;
  *)
    echo "[install_onnxruntime] unknown variant '${VARIANT}' (expected: cpu|gpu)" >&2
    exit 1
    ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.tgz"
echo "[install_onnxruntime] installing ONNX Runtime ${VERSION} (${VARIANT}) -> ${DEST}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fSL "${URL}" -o "${TMP_DIR}/ort.tgz"
else
  wget -q "${URL}" -O "${TMP_DIR}/ort.tgz"
fi

tar -xzf "${TMP_DIR}/ort.tgz" -C "${TMP_DIR}"

rm -rf "${DEST}"
mv "${TMP_DIR}/${PKG}" "${DEST}"

# The C++ build and the emotiefflib patch expect the shared libraries in lib64.
mkdir -p "${DEST}/lib64"
if compgen -G "${DEST}/lib/*.so*" >/dev/null; then
  mv "${DEST}"/lib/*.so* "${DEST}/lib64/"
fi

# onnxruntime's CMake package file is referenced as ONNXRuntimeConfig.cmake by
# find_package(onnxruntime); the release only ships the lowercase name.
CMAKE_DIR="${DEST}/lib/cmake/onnxruntime"
if [ -f "${CMAKE_DIR}/onnxruntimeConfig.cmake" ] && [ ! -f "${CMAKE_DIR}/ONNXRuntimeConfig.cmake" ]; then
  cp "${CMAKE_DIR}/onnxruntimeConfig.cmake" "${CMAKE_DIR}/ONNXRuntimeConfig.cmake"
fi

if [ "${VARIANT}" = "gpu" ]; then
  if [ ! -f "${DEST}/lib64/libonnxruntime_providers_cuda.so" ]; then
    echo "[install_onnxruntime] ERROR: GPU build has no libonnxruntime_providers_cuda.so" >&2
    exit 1
  fi
  echo "[install_onnxruntime] GPU variant also needs CUDA 12 + cuDNN 9 on the loader path"
fi

echo "[install_onnxruntime] done: ${DEST} ($(ls "${DEST}/lib64" | grep -c '\.so' ) shared libraries)"
