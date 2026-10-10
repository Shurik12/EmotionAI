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

## ONNX Runtime GPU (RTX 3060)

Ran on the host (12 cores) with `onnxruntime-linux-x64-gpu-1.21.0` + CUDA 12 +
cuDNN 9 and the **CUDAExecutionProvider** appended (the shipped EmotiEffLib ONNX
backend uses default/CPU providers, so the provider must be appended explicitly).

| Backend | host | ms/frame | × realtime |
|---|---|---:|---:|
| torch CPU | container, 8 CPUs | 63.2 | 78× |
| onnx CPU | host, 12 cores | 57.5 | 86× |
| **onnx CUDA** | host, 12 cores + RTX 3060 | **39.7** | **125×** |

Average over all 4 videos with CUDA: **39.7 ms/frame, 25.2 fps, 125.5×**, stable
across repeats (±1%). GPU utilisation was only **~5–11 %** and VRAM ~1.2 GB, so
inference is no longer the bottleneck — **CPU decode + MTCNN detection now
dominate**. The GPU still needs a CUDA build of ORT/libtorch and a Docker runtime
with GPU access (the current Compose setup has none).

## Resource requirements for a target speed

| Target | CPU | RAM | GPU |
|---|---|---|---|
| ~20× realtime | ~2 cores | ~3.5 GiB | — |
| ~40× realtime | ~4 cores | ~3.5 GiB | — |
| ~80× realtime | ~8 cores | ~3.5 GiB | — (current container config) |
| ~125× realtime | ~8–12 cores | ~3.5 GiB | RTX 3060, ~1.2 GB VRAM (onnx CUDA) |
| ~160×+ | ~16 cores | ~3.5 GiB | — or a bigger GPU (decode-limited above ~125×) |

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
3. **GPU (validated)** — `onnxruntime-gpu` + CUDA 12 + cuDNN 9 with the
   CUDAExecutionProvider appended measured **125×** on the RTX 3060. Needs a
   Docker runtime with GPU access; above ~125× the CPU decoder becomes the limit.
4. **Reduce work** — larger `frame_interval_seconds`, lower capture resolution,
   or a cheaper face detector.
