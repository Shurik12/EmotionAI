# AGENTS.md — EmotionAI agent briefing

Operational context for AI agents and new contributors. Read this first; it captures what is
expensive or impossible to rediscover by reading code. Human-facing docs live in [README.md](README.md).

## What this is

Multimodal emotion recognition service: facial emotion in image/video, emotion from speech, and
burnout risk scored against a per-user baseline. **Single C++23 binary** with a hand-written
non-blocking `epoll` HTTP server — no web framework, and **no Python at runtime**. Python exists
only for training and for exporting model weights to C++.

Repo: `git@github.com:Shurik12/EmotionAI.git`, branch `main`. No CI is configured.

## Commands that actually work

```bash
make install          # apt deps + submodules + libtorch/onnxruntime into contrib/ + emotiefflib patch
make python_env       # venv + requirements.txt
make build            # configure (CMake+Ninja) + build_backend + build_frontend
make up               # docker compose: dragonfly + server
./build/emotionai     # run binary directly (from repo root only)
```

### Broken — do not trust these

| Command | Why it fails |
|---|---|
| `make run` | Listed in `.PHONY` but **no such target exists**. Use `make up` or run the binary. |
| `make unit_tests` / `make integration_tests` | Point at `build/tests/EmotionAI_*Tests`, but `add_subdirectory(tests)` is **commented out** in `CMakeLists.txt:211-213`. The binaries are never built. Uncomment (and `enable_testing()` at line 209) before attempting to run tests. |
| `make models` | Runs `cd venv && python3 prepare_models_for_emotieffcpplib.py`, but the script is at `contrib/emotiefflib/models/`. The Makefile defines an unused `MODELS_DIR` with the correct path. Run manually: `. venv/bin/activate && cd contrib/emotiefflib/models && python3 prepare_models_for_emotieffcpplib.py` |
| `make clean` | Help text claims it removes venv and caches. It references an **undefined** `$(VENV_DIR)`, and targets `frontend/build` + root `package-lock.json`, neither of which exists (real dir is `frontend/dist`). |
| `install_deps.sh` | Does **not** install `libfftw3-dev` or FFmpeg dev packages. FFTW3 is mandatory — a clean install fails at build. Also its "Verifying installations" output claims to check PostgreSQL client, httplib and redis-plus-plus, none of which it installs. |

Install the missing deps manually:

```bash
sudo apt-get install -y libfftw3-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
```

## Non-obvious facts

**A fresh clone cannot build.** `contrib/libtorch/`, `contrib/onnxruntime/`, `contrib/emotiefflib/`
and `models/` are all gitignored, as is `config.yaml`. Run `make install` first, then
`cp config_template.yaml config.yaml`.

**Config loading is hardcoded.** `main.cpp:32` calls `loadFromFile("config.yaml")` — a relative
path resolved against the **current working directory**, not the binary location. There is no CLI
argument parsing. `docker-compose.yml` sets `CONFIG_FILE=/emotionai/config.yaml` but **no code ever
reads that variable**; Docker works only because the container cwd happens to be `/emotionai`.

**Inference backend is chosen twice.** Compile time: `HAVE_TORCH` / `HAVE_ONNX` are defined based on
whether `contrib/libtorch` / `contrib/onnxruntime` exist (`CMakeLists.txt:114-123`). Runtime:
`model.backend` in `config.yaml`. Both must agree.

**FFmpeg is optional, FFTW3 is not.** Without FFmpeg the build succeeds with a CMake warning and
audio is silently limited to WAV — MP3/MP4 uploads will fail at runtime, not build time.

**Emotion label count depends on the model filename.** `Image.cpp:216-219`: if the configured model
is `enet_b2_7.pt` you get 7 classes; otherwise 8 (including `contempt`). Changing the model path
changes the response schema.

**Burnout results are translation keys, not text.** `BurnoutAnalyzer` returns keys like
`burnout.level.high` which the frontend resolves via `frontend/src/utils/translations.js` (RU/EN).
Never emit human-readable strings from the C++ side.

**`src/common/httplib.h` is vendored but is not the request path.** Routing happens in the epoll
loop in `src/server/Server.cpp`. Don't add handlers to httplib expecting them to be reachable.

### Adding an API route requires up to three edits

1. Register in `post_routes_` or `get_routes_` — `Server.cpp:316-349`
2. Add to the `options_routes_` list for CORS preflight — `Server.cpp:352-358`
3. If parameterized (`/api/foo/<id>`), add prefix matching in the GET dispatcher — `Server.cpp:626-649`.
   Exact-match maps are checked first; prefix routes are hardcoded `if/else` on `path.find(...)`.

Missing step 2 or 3 fails silently as a 404 or a CORS error in the browser.

## Code conventions

The codebase is **inconsistent** — match the file you are editing rather than imposing one style.

- **Header guards:** `#pragma once` dominant (25 of 35 headers); a few use `#ifndef` guards.
- **Member naming:** mixed. Trailing underscore (`epoll_fd_`, `post_routes_`, `host_`) in `server/`,
  `config/`, `db/`; `m_` prefix (`m_config`, `m_accessToken`) in `gigachat/`, `emotionai/`, `client/`.
- **Namespaces:** `audio`, `emotionai`, `db` are used; `server/` and `config/` classes are largely
  in the global namespace. `namespace fs = std::filesystem` is the standard alias.
- **Logging:** use the `LOG_INFO` / `LOG_ERROR` / `LOG_WARN` / `LOG_DEBUG` macros (415 uses).
  `Logger::instance()` direct calls appear only 11 times, mostly in `main.cpp` before init.
- **JSON:** `nlohmann/json`. **Config:** yaml-cpp behind the typed accessors in `src/config/Config.h`.
- **Singletons:** `Config::instance()`, `Logger::instance()`.
- **Tests:** GoogleTest + gmock. `tests/mocks/MockEmotiEffLib.h` allows tests to run without model
  weights. Test configs live in `tests/configs/` on ports 8081–8083 and Redis DBs 1–3 so they never
  collide with a dev server.

## Layout map

| Path | Contents |
|---|---|
| `src/server/` | epoll HTTP server, routing, `ThreadPool` |
| `src/emotionai/` | `Image`, `Audio`, `FileProcessor` — inference orchestration |
| `src/mtcnn/` | face detection (pnet/rnet/onet) |
| `src/audio/` | `LibrosaFeatureExtractor` (FFTW3-based), `BurnoutAnalyzer` |
| `src/db/` | `RedisManager`, `DragonflyManager`, `TaskManager` |
| `src/cluster/` | `ClusterManager`, `DistributedTaskManager` |
| `src/storage/` | `FileStorage` interface + Local / NFS / S3 (MinIO) |
| `src/metrics/` | Prometheus collector and middleware |
| `frontend/src/` | React 18 + Vite 5 SPA; `api/`, `components/`, `hooks/`, `context/`, `utils/` |
| `config/` | YAML profiles, `nginx` vhost template, `service` systemd unit |
| `docker/` | five compose variants, one per deployment topology |
| `training/` | PyTorch training and dataset prep |
| `tests/` | unit / integration / end_to_end / benchmark |

## Guardrails

- **Never commit** `config.yaml`, `models/`, `venv/`, `build/`, `uploads/`, `results/`, `logs/`,
  `data/`, `frontend/dist/` — all gitignored, and `config.yaml` holds credentials
  (`dragonfly.password`, `gigachat.auth_key`).
- **`contrib/emotiefflib` is a submodule and is currently dirty** (`git status` shows
  ` m contrib/emotiefflib`). Inspect before staging; don't commit a stray submodule pointer bump.
- `requirements.txt` is for **training/export only**. Adding a package there does not make it
  available to the running server.
- The systemd unit `config/service` will not start as written: `ExecStart` points at
  `build/EmotionAI` (actual binary is `build/emotionai`) and passes `--config`, which the binary
  does not parse.
- Default bind is `0.0.0.0:80`, which needs root or a `cap_net_bind_service` capability.
- A stray empty directory named `mkdir/` sits in the repo root — an artifact, not a real module.
