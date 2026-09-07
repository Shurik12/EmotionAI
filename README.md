# EmotionAI

Multimodal emotion recognition service. Detects facial emotions in images and video, classifies
emotion from speech, and scores burnout risk by comparing a recording against a stored personal
baseline.

The backend is a single C++23 binary with a self-contained HTTP server — no web framework, no
Python at runtime. Python is used only for model training and for exporting model weights.

---

## Features

- **Face emotion recognition** — MTCNN face detection (P-Net / R-Net / O-Net) feeding an
  EmotiEffLib classifier. 8 classes (`anger`, `contempt`, `disgust`, `fear`, `happiness`,
  `neutral`, `sadness`, `surprise`), or 7 classes with `enet_b2_7.pt`.
- **Audio emotion recognition** — wav2vec2 / WavLM acoustic models, plus a native C++
  reimplementation of the Librosa feature pipeline (mel spectrogram, 13 MFCCs, pitch, intensity,
  pause structure, speech rate, voice activity) built on FFTW3.
- **Burnout analysis** — compares a current recording against a per-user baseline across five
  weighted components: emotional exhaustion (0.25), prosodic flattening (0.25), pause tempo (0.20),
  negative activation (0.15), positive affect loss (0.15). History-aware, returns translation keys
  so the frontend localises the verdict.
- **External influence (scam) signal** — pilot anti-fraud aid: splits one call recording into
  consecutive ≤10 s windows and flags a *sustained* combination of emotional tension and two-sided
  speech-behavior change (tempo/pauses/prosody deviating in either direction) across ≥3 fragments,
  optionally confirmed by context flags (urgency, coached answers, "safe account", …). Emits a
  severity status + manager action — **not** fraud proof, never blocks an operation. All weights
  and boundaries live in the `external_influence:` config section for recalibration.
- **Batch and realtime pipelines** — two Dragonfly/Redis queues with visibility timeouts and
  retries, so long video jobs never block quick single-image requests.
- **Pluggable storage** — local, NFS, or S3 (MinIO) behind one `FileStorage` interface, selected by
  config.
- **Horizontal scaling** — stateless server instances behind nginx, coordinated through a shared
  Dragonfly cluster and distributed task manager.
- **Observability** — Prometheus metrics endpoint with provisioned Grafana dashboards.
- **Optional GigaChat enrichment** — LLM-generated commentary on detected emotions, gated behind a
  confidence threshold.
- **React frontend** — Vite build, react-router, multilingual (RU/EN), cookie consent, charts.

Accepted uploads: `png`, `jpg`, `jpeg`, `mp4`, `avi`, `webm`, `mp3`, `wav` (50 MB default limit).

> Audio formats other than WAV require FFmpeg at build time. Without it the server still runs but
> only accepts WAV input.

---

## Architecture

```
                        ┌──────────────────┐
        browser ───────▶│  nginx           │  TLS, static, reverse proxy
                        └────────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
      │ emotionai (A) │  │ emotionai (B) │  │ emotionai (N) │   stateless, epoll
      └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
              └──────────────────┼──────────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │  Dragonfly cluster  │  task queues + shared state
                      └──────────┬──────────┘
                                 ▼
                      ┌─────────────────────┐
                      │  shared storage     │  local / NFS / S3
                      └─────────────────────┘
```

The HTTP layer is hand-rolled on POSIX sockets with a non-blocking `epoll` event loop
(`src/server/Server.cpp`). One process multiplexes all connections; a `ThreadPool` offloads
inference work so the event loop never stalls.

### Project layout

```
src/
├── server/      epoll HTTP server, routing, thread pool
├── emotionai/   Image / Audio / FileProcessor inference orchestration
├── mtcnn/       face detection (pnet, rnet, onet)
├── audio/       acoustic feature extraction, burnout + external-influence analyzers
├── db/          Redis + Dragonfly managers, TaskManager
├── cluster/     ClusterManager, DistributedTaskManager
├── storage/     FileStorage interface + Local / NFS / S3 backends
├── metrics/     Prometheus collector and middleware
├── gigachat/    optional LLM client
├── config/      typed config loaded from YAML
├── logging/     spdlog wrapper
├── client/      HTTP client for tests and benchmarks
└── common/      base64, uuid, httplib, librosa helpers

frontend/        React 18 + Vite 5 SPA
training/        PyTorch training and dataset prep scripts
tests/           unit, integration, end-to-end, benchmarks
config/          YAML profiles, nginx vhost, systemd unit
docker/          compose variants for each deployment topology
monitoring/      prometheus.yml + Grafana provisioning
contrib/         emotiefflib, minio-cpp, inih (submodules), libtorch, onnxruntime
```

---

## Requirements

**System packages** (Ubuntu/Debian): build-essential, cmake, ninja-build, pkg-config, nginx,
certbot, python3.12-venv, npm, redis-server, gdb, and dev libraries for OpenCV, yaml-cpp, spdlog,
fmt, nlohmann-json, libcurl + curlpp, OpenSSL, zlib, pugixml, hiredis.

**Also required by the build but not installed by `install_deps.sh`** — add these manually:

```bash
sudo apt-get install -y libfftw3-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
```

FFTW3 is mandatory (mel spectrogram). FFmpeg is optional but needed for MP3/MP4 audio.

**Vendored into `contrib/`** by the install script:

| Component   | Version     | Purpose                          |
|-------------|-------------|----------------------------------|
| libtorch    | 2.1.0 (CPU) | inference backend (`HAVE_TORCH`)  |
| onnxruntime | 1.21.0      | inference backend (`HAVE_ONNX`)   |
| emotiefflib | submodule   | emotion model definitions         |
| minio-cpp   | submodule   | S3 storage                        |
| inih        | submodule   | INI parsing                       |

Inference backend is chosen at **compile time** by which of `contrib/libtorch` / `contrib/onnxruntime`
exist, and at **runtime** by `model.backend` in `config.yaml`.

---

## Quick start

```bash
# 1. System deps, git submodules, contrib downloads, emotiefflib patch
make install

# 2. Python venv + requirements
make python_env

# 3. Export C++ model headers from the Python scripts
make models

# 4. Configure
cp config_template.yaml config.yaml
$EDITOR config.yaml

# 5. Build (CMake + Ninja, then Vite frontend)
make build

# 6a. Run under Docker (server + Dragonfly)
make up

# 6b. ...or run the binary directly (needs a reachable Redis/Dragonfly)
./build/emotionai
```

The binary reads `config.yaml` from its **current working directory**, not from `build/`. The path is
hardcoded in `main.cpp`, so start the binary from the repo root. In Docker this works because the
container's working directory is `/emotionai`, where `config.yaml` is mounted.

> The compose file exports a `CONFIG_FILE` variable, but the binary does not currently read it.

`make help` lists every target.

---

## Configuration

Profiles live in `config/`; copy the one matching your topology over the root `config.yaml`:

| Profile                     | Topology                                      |
|-----------------------------|-----------------------------------------------|
| `config_single_local.yaml`  | one instance, local disk storage              |
| `config_single_s3.yaml`     | one instance, S3/MinIO storage                |
| `config_all.yaml`           | clustered, distributed queue, shared storage  |

Key sections (see `config_template.yaml` for full annotated defaults):

| Section             | Controls                                                        |
|---------------------|-----------------------------------------------------------------|
| `server`            | bind host and port (default `0.0.0.0:80`)                       |
| `paths`             | upload / results / logs / frontend directories                  |
| `app`               | max upload size, allowed extensions, task and result TTLs       |
| `logging`           | level, rotation size, file count, console/file patterns         |
| `queue`             | batch and realtime queue names, visibility timeout, retries     |
| `task_management`   | cache TTL and size, batching window                             |
| `dragonfly`         | host, port, db index, password, pool size, pipelining           |
| `storage`           | `local` / `nfs` / `s3` plus base path or bucket credentials     |
| `mtcnn`             | min face size, post-processing, keep-all, device                |
| `model`             | backend (`torch`/`onnx`), emotion model, audio model, det. path |
| `cluster`           | enable distributed coordination                                 |
| `gigachat`          | optional LLM enrichment, auth key, min confidence               |
| `external_influence`| all weights/boundaries of the scam-signal analyzer (pilot)      |

Unit, integration and e2e tests use separate configs in `tests/configs/` on ports 8081–8083 and
Redis DBs 1–3, so they never collide with a running dev server.

---

## API

All endpoints are under `/api`. CORS preflight (`OPTIONS`) is handled for every route.

### POST

| Endpoint                  | Description                                            |
|---------------------------|--------------------------------------------------------|
| `/api/upload`             | Queue an image/video/audio file for batch processing   |
| `/api/upload_realtime`    | Process a file immediately on the realtime queue       |
| `/api/upload_burnout`     | Queue audio for burnout analysis                       |
| `/api/submit_application` | Submit a contact/access application                    |
| `/api/batch_progress`     | Progress for a set of task ids                         |
| `/api/burnout/analyze`    | Analyze current sample against a supplied baseline     |
| `/api/burnout/baseline`   | Create or update a user's burnout baseline             |
| `/api/upload_external_influence` | Queue a call recording for external influence analysis |
| `/api/external-influence/analyze` | Re-run the external influence status with context flags |

### GET

| Endpoint                      | Description                                  |
|-------------------------------|----------------------------------------------|
| `/api/health`                 | Liveness/readiness probe                     |
| `/api/progress/<task_id>`     | Status of a single queued task               |
| `/api/results/<filename>`     | Fetch a completed result payload             |
| `/api/storage/info`           | Active storage backend and capacity info     |
| `/api/metrics`                | Prometheus metrics                           |
| `/api/burnout/baseline/<user_id>` | Retrieve a stored baseline               |
| `/static/<path>`              | Uploaded and generated assets                |

Any path that is not an API route falls through to the React SPA, so client-side routing works.

```bash
curl http://localhost/api/health
curl -F "file=@sample.jpg" http://localhost/api/upload_realtime
curl http://localhost/api/progress/<task_id>
```

---

## Deployment

### Docker

`docker-compose.yml` at the repo root runs Dragonfly plus the server on the host network with
8 GB / 2 CPU limits and an `/api/health` healthcheck. Prebuilt topology variants are in `docker/`:

| File                                 | Use case                          |
|--------------------------------------|-----------------------------------|
| `docker-compose-single-local.yml`    | single node, local storage        |
| `docker-compose-single-s3.yml`       | single node, S3 storage           |
| `docker-compose-replicas.yml`        | replicated stateless instances    |
| `docker-compose-distributed.yml`     | distributed queue across nodes    |
| `docker-compose-all.yaml`            | full stack incl. monitoring       |

Copy the variant you want over the root file, then `make up` (`make up-build` to rebuild images).

### Bare metal (nginx + systemd)

Ready-made templates are in `config/`:

```bash
sudo cp config/nginx   /etc/nginx/sites-available/emotion-ai
sudo cp config/service /etc/systemd/system/emotion-ai.service
sudo ln -s /etc/nginx/sites-available/emotion-ai /etc/nginx/sites-enabled/
sudo systemctl daemon-reload
sudo systemctl enable --now emotion-ai.service
sudo systemctl reload nginx
```

The systemd unit hardens with `LimitNOFILE=65536`, `MemoryMax=4G`, `CPUQuota=200%`, and logs to
syslog under the `emotion-ai` identifier. Edit `WorkingDirectory` and `ExecStart` to match your
checkout before enabling it.

### TLS

```bash
sudo systemctl stop nginx
sudo certbot renew
sudo systemctl start nginx
sudo certbot certificates
```

### Monitoring

`monitoring/prometheus.yml` plus Grafana datasource and dashboard provisioning under
`monitoring/grafana/`. The server exposes metrics at `/api/metrics`.

---

## Testing

```bash
make test   # configures with -DBUILD_TESTS=ON, builds and runs build/tests/emotionai_tests
```

> Tests are currently disabled in the build: `add_subdirectory(tests)` is commented out at
> `CMakeLists.txt:211-213`, so `make test` fails until that block and `enable_testing()` on line 209
> are restored. `option(BUILD_TESTS ... ON)` on line 4 is otherwise inert. GTest and gmock are
> required (`sudo apt-get install -y libgtest-dev libgmock-dev`).

- `tests/unit/` — config, db, logging
- `tests/integration/` — server and end-to-end flows
- `tests/benchmark/` — concurrent client, load and inference benchmarks
- `tests/mocks/` — `MockEmotiEffLib` so tests run without model weights
- `tests/fixtures/` — sample images

Frontend:

```bash
cd frontend && npm test && npm run lint
```

---

## Training

```bash
cd training
cp /path/to/enet_b0_8_va_mtl.pt ./pretrained_model.pt
python3 process_images.py        # raw images -> processed_images/
python3 create_train_tsv.py      # build train.tsv / test.tsv manifests
python3 train_multitask.py       # multitask training
python3 robust_optimization.py   # hyperparameter search (hyperopt)
```

Audio models (`wav2vec2-emotion-recognition`, `wavlm-emotion-russian-resd`) are trained in their
own repositories and exported to `models/audio_model.pt`.

---

## Troubleshooting

**`Failed to load config, using defaults`** — the binary cannot find `config.yaml` in the working
directory. Start it from the repo root or set `CONFIG_FILE`.

**Build fails on `fftw3.h`** — install `libfftw3-dev`.

**MP3/MP4 audio rejected** — FFmpeg was missing at build time. Install the dev packages and
reconfigure with `make configure`.

**`Cannot connect to Dragonfly`** — start it: `make up`, or
`docker run -d -p 6379:6379 --name dragonfly docker.dragonflydb.io/dragonflydb/dragonfly`.

**Port 80 already in use** — change `server.port` in `config.yaml`, or let nginx terminate on 80
and run the binary on an internal port.

---

## Contributing

1. Open an issue: https://github.com/Shurik12/EmotionAI/issues
2. Branch from `main`, keep changes focused, add tests for new behaviour.
3. Run `make test` before opening a PR.

## License

See [LICENSE](LICENSE).
