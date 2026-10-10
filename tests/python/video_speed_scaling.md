# Video processing: fps vs CPU / RAM / VRAM (EMO-23)

Machine: 12 vCPU, 31 GiB RAM, NVIDIA RTX 3060 (12 GB), driver 580.
Video: `Светлана.mp4` — 1280×720, 25 fps, ~0.5 Mbps; 175 sampled frames
(1 frame / 5 s, `video.frame_interval_seconds: 5.0`).
Each run = upload + full processing; resources sampled every 1 s while running.

## CPU-only (backend `torch`, 8-model pool)

| CPUs | fps | ms/frame | × realtime | peak RAM | peak CPU |
|---:|---:|---:|---:|---:|---:|
| 2  | 3.9  | 256 | 19× | 3.14 GiB | 205% |
| 4  | 8.3  | 121 | 41× | 3.30 GiB | 403% |
| 6  | 12.4 | 81  | 62× | 3.36 GiB | 606% |
| 8  | 15.8 | 63  | 78× | 3.41 GiB | 808% |
| 12 | 19.3 | 52  | 96× | 3.45 GiB | 1082% |

fps scales roughly linearly to ~8 cores (≈ **1.6 fps per core**, ≈ 10×
realtime per core), then flattens as decode/MTCNN saturate the cores.

## GPU (backend `onnx` + CUDA, RTX 3060); CPU pinned with `taskset`

| CPUs | fps | ms/frame | × realtime | peak VRAM | peak GPU util | peak RAM |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 9.7  | 103 | 48×  | 1.38 GiB | 6%  | 3.10 GiB |
| 4  | 17.4 | 58  | 86×  | 1.38 GiB | 11% | 3.07 GiB |
| 6  | 21.7 | 46  | 108× | 1.38 GiB | 11% | 3.06 GiB |
| 8  | 24.9 | 40  | 123× | 1.38 GiB | 12% | 3.06 GiB |
| 12 | 24.8 | 40  | 123× | 1.38 GiB | 6%  | 3.08 GiB |

VRAM includes the ~0.25 GiB driver baseline; the model adds ~1.1 GiB. GPU
utilisation stays ≤ 12% — inference is **not** the bottleneck, CPU decode + MTCNN
are, so the GPU path plateaus at ~25 fps once ~8 CPU cores are available.

## CPU-only vs GPU at equal CPU (fps)

| CPUs | CPU-only | GPU | speedup |
|---:|---:|---:|---:|
| 2  | 3.9  | 9.7  | 2.5× |
| 4  | 8.3  | 17.4 | 2.1× |
| 6  | 12.4 | 21.7 | 1.8× |
| 8  | 15.8 | 24.9 | 1.6× |
| 12 | 19.3 | 24.8 | 1.3× |

## Conclusions

- **CPU** drives fps almost linearly to ~8 cores, then saturates (decode + face
  detection dominate). ~20× realtime (≈8 fps) needs ~4 cores; ~80× needs ~8
  cores; ~120×+ needs GPU or many cores.
- **GPU** helps most when CPU is scarce (2.5× at 2 cores) and less as CPU grows
  (1.6× at 8). Above ~8 cores the GPU path is decode-bound and plateaus ~25 fps.
- **RAM** is flat at ≈3.1–3.5 GiB regardless of configuration (8-model pool +
  server); it is not a speed driver. The container limit of 8 GiB is ample.
- **VRAM** is ≈1.1 GiB for the model (1.38 GiB total including the driver) and
  constant, so any modern CUDA GPU suffices; the model is far too small to be
  GPU-bound.

## Sizing guide

| Target | Recommended |
|---|---|
| ~20–40× realtime | 2–4 CPU cores, no GPU |
| ~80× realtime | 8 CPU cores, no GPU |
| ~120× realtime | 8 CPU cores **+** any CUDA GPU (onnx) |
| >125× realtime | GPU won't help further — decode-bound; add CPU / hardware decode |
