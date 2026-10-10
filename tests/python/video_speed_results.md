# Video processing performance (EMO-23)

Machine: 12 vCPU x86_64, 31 GiB RAM, NVIDIA RTX 3060 (see GPU note).
Videos in `videos/Видео_Исследование/`: h264, 1280×720, 25 fps, ~0.5 Mbps.
Pipeline samples 1 frame every 5 s (`video.frame_interval_seconds`).
Measured with `tests/python/measure_video_speed.py`.

## Throughput vs CPU (torch, `Светлана.mp4`, 175 frames)

| CPUs | ms/frame | frames/s | × realtime |
|-----:|---------:|---------:|-----------:|
| 2    | 256      | 3.9      | 19×        |
| 4    | 121      | 8.3      | 41×        |
| 8    | 63       | 15.8     | 78×        |

Near-linear scaling to 8 cores → ~**10× realtime per core** for this profile.
(2 cores was the old `docker-compose` cap; the workload was purely throttled.)

## Full benchmark (8 CPUs, all 4 videos)

| File | Duration | Frames | Processing | ms/frame | × realtime |
|---|---:|---:|---:|---:|---:|
| video1300205474.mp4 | 2025.8 s | 407 | 26.2 s | 64.3 | 77.4× |
| video1544216812.mp4 | 2198.3 s | 441 | 27.2 s | 61.6 | 80.9× |
| Вера.mp4 | 1333.6 s | 268 | 17.1 s | 63.8 | 78.0× |
| Светлана.mp4 | 867.9 s | 175 | 11.1 s | 63.2 | 78.4× |
| **Average** | | | **20.4 s** | **63.2** | **78.7×** |

Before EMO-23: 209.8 ms/frame, 4.8 fps, 24× → now ~**3.3× faster**.

## ONNX vs Torch (8 CPUs, `Светлана.mp4`)

| Backend | ms/frame | × realtime |
|---|---:|---:|
| torch (`enet_b0_8_va_mtl.pt`) | 63.2 | 78.4× |
| onnx (`enet_b0_8_va_mtl.onnx`, ORT 1.21) | 68.9 | 72.0× |

On this CPU the ONNX Runtime build is ~10% **slower** than libtorch for this
model, so the backend switch is not a CPU win here.

## Resource requirements for a target speed

| Target | CPU | RAM | Notes |
|---|---|---|---|
| ~20× realtime | ~2 cores | ~3.5 GiB | |
| ~40× realtime | ~4 cores | ~3.5 GiB | |
| ~80× realtime | ~8 cores | ~3.5 GiB | current config |
| ~160× realtime | ~16 cores, or GPU | | scales linearly on CPU |

- **CPU:** the bottleneck (H.264 decode + MTCNN + ENet inference). ~1 core per
  ~10× realtime for 720p/25fps at 1 frame / 5 s.
- **RAM:** ~3.5 GiB resident (8-model pool + server); 8 GiB container limit is
  comfortable.
- **GPU:** an RTX 3060 is present, but the shipped libtorch/onnxruntime are
  **CPU-only** (no `libtorch_cuda.so`, no ORT CUDA provider), so the GPU is
  currently unused. A CUDA build would likely give a large speedup for the
  inference stage; decode stays CPU unless a hardware decoder is used.

## Scaling options

1. **More CPU cores** — near-linear (see table).
2. **Horizontal replicas** — `docker/docker-compose-replicas.yml` runs several
   servers behind nginx.
3. **GPU build** — CUDA libtorch (`WITH_TORCH` → CUDA build) or
   `onnxruntime-gpu`; would accelerate inference.
4. **Reduce work** — larger `frame_interval_seconds`, lower capture resolution,
   or a cheaper face detector.
