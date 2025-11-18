# EmotionAI Server Benchmarks

This directory contains performance benchmarks for the EmotionAI emotion recognition server using Google Benchmark.

## Prerequisites

- Google Benchmark library
- Running EmotionAI server on localhost:8080
- Test assets (image and video files)

## Test Assets

Place the following test files in the `tests/benchmark/assets/` directory:

- `test_image.jpg` - A sample image for testing
- `test_video.mp4` - A sample video for testing

## Available Benchmarks

### 1. Single Request Benchmarks
- `BM_ImageUploadAndProcessing` - Single image upload and processing
- `BM_VideoUploadAndProcessing` - Single video upload and processing

### 2. Concurrent Load Benchmarks
- `BM_ConcurrentImageUploads` - Multiple concurrent image uploads (configurable)
- `BM_ConcurrentVideoUploads` - Multiple concurrent video uploads (configurable)
- `BM_MixedWorkload` - Mixed image/video workload (70/30 ratio)

### 3. API Performance Benchmarks
- `BM_TaskStatusPolling` - Performance of individual task status checks
- `BM_BatchProgressChecking` - Performance of batch progress checks

## Building and Running

```bash
# Build benchmarks
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make emotionai_benchmark

# Run all benchmarks
./tests/benchmark/emotionai_benchmark

# Run specific benchmark with custom options
./tests/benchmark/emotionai_benchmark --benchmark_filter="BM_ConcurrentImageUploads"

# Run with detailed output
./tests/benchmark/emotionai_benchmark --benchmark_format=json > results.json