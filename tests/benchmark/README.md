# EmotionAI Server Benchmarks

This directory contains performance benchmarks for the EmotionAI emotion recognition server using Google Benchmark.

## Prerequisites

- Google Benchmark library
- Running EmotionAI server on `localhost:8080` (or set `FocusBenchmark::Config::server_host` / `server_port`)
- Test assets (image and video files)

## Test Assets

Place the following test files in the `tests/benchmark/assets/` directory:

- `test_face.jpg` - A sample image for Focus camera frame uploads
- `test_image.jpg` - A sample image for `/api/upload` benchmarks
- `test_video.mp4` - A sample video for `/api/upload` benchmarks

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

### 4. Focus Camera Benchmarks (EMO-17)

The Focus page is a live camera stream: the browser creates a session, posts
JPEG frames at ~1 fps, polls the session for accumulated observations, and
closes the session when the user leaves. The server scores each frame on the
thread pool (face detection + emotion model) and returns `202` immediately
after queueing, so the HTTP path is the hot spot.

- `BM_FocusSingleSession` - One camera client running the full session flow
  (create → upload 10 frames → poll → close) in a loop for 3 s. Baseline for
  the ~1 fps real-world case.
- `BM_FocusConcurrentSessions` - N concurrent camera clients (1/4/8/16/32) for
  3 s each. Measures **frame-upload RPS**, the unit that matters for the
  camera. Each client runs the full session flow.
- `BM_FocusPolling` - N concurrent clients polling `GET /api/focus/session/<id>`
  for 1 s. Measures the maximum poll RPS the page can sustain.
- `BM_FocusFrameUpload` - N concurrent clients blasting frames through one
  shared session for 1 s. Isolates the hottest endpoint
  (`POST /api/focus/session/<id>/frame`).

All four report `RPS_frames` / `RPS_polls` as a rate counter (requests/sec,
auto-scaled to k/s).

## Building and Running

```bash
# Build benchmarks
make benchmark

# Run all benchmarks
./build/tests/benchmark/emotionai_benchmark

# Run only the Focus benchmarks
./build/tests/benchmark/emotionai_benchmark --benchmark_filter="BM_Focus"

# Run a specific Focus benchmark
./build/tests/benchmark/emotionai_benchmark --benchmark_filter="BM_FocusConcurrentSessions/16"

# Run with detailed output
./build/tests/benchmark/emotionai_benchmark --benchmark_format=json > results.json

# Run Focus benchmarks and save a measurement to measurements/
./tests/benchmark/run_focus_benchmark.sh [host] [port]
```

> Note: the Focus benchmarks require a running server. If the server is not
> reachable they will be reported as skipped (not failed), so the suite still
> exits cleanly in CI without a live server.

## Interpreting the Focus RPS numbers

The Focus camera is designed for **1 frame per second** per user, so the
server needs to sustain only a few RPS in normal operation. The benchmarks
here measure head-room:

| Benchmark | Typical RPS (16-core box, DEBUG build) |
|---|---|
| `BM_FocusFrameUpload` (1–16 clients) | ~370–550 frames/s |
| `BM_FocusConcurrentSessions` (16 clients, full flow) | ~150 frames/s |
| `BM_FocusPolling` (1–16 clients) | ~1.5–1.7 polls/s |

The frame-upload endpoint is the bottleneck; the full session flow adds the
cost of session create/poll/close. A 10 RPS requirement is met with a wide
margin even on a single core.

## Storing measurement results

`run_focus_benchmark.sh` writes a timestamped JSON file into
`measurements/` (e.g. `measurements/focus-20261014-204227.json`) and prints a
human-readable summary. Measurement files are committed to the repository so
that performance regressions are visible in code review. See
[measurements/README.md](measurements/README.md) for details.
