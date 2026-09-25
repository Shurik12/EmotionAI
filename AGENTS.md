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
make help             # self-documenting target list, generated from the Makefile
make install          # apt deps + submodules + libtorch/onnxruntime into contrib/ + emotiefflib patch
make python_env       # venv + requirements.txt
make models           # export C++ model headers from contrib/emotiefflib/models
make build            # configure (CMake+Ninja) + build_backend + build_frontend
make build_harness    # build tools/burnout_harness (offline burnout measurement)
make up               # docker compose: dragonfly + server
./build/emotionai     # run binary directly (from repo root only)
```

### Broken — do not trust these

| Command | Why it fails |
|---|---|
| `make run` | **No such target.** Use `make up` or run `./build/emotionai` from the repo root. |
| `make test` | Correctly targets the single real binary `build/tests/emotionai_tests` and passes `-DBUILD_TESTS=ON`, but `add_subdirectory(tests)` is **commented out** in `CMakeLists.txt:211-213`, so the target is never generated. Uncomment that block plus `enable_testing()` on line 209. Note `tests/CMakeLists.txt` builds **one** binary, `emotionai_tests` — there is no unit/integration split despite what older docs claimed. |
| `install_deps.sh` | Does **not** install `libfftw3-dev` or FFmpeg dev packages. FFTW3 is mandatory — a clean install fails at build. It also omits `libgtest-dev`, needed for `make test`. Its "Verifying installations" output claims to check PostgreSQL client, httplib and redis-plus-plus, none of which it installs. |

Install the missing deps manually:

```bash
sudo apt-get install -y libfftw3-dev libgtest-dev libgmock-dev \
  ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
```

### Shell permissions (agent command hygiene)

Bash `allow` rules are prefix-anchored and checked against **every top-level segment** of a
command (split on `&&`, `;`, `|`); a chain is auto-approved only if each segment matches a rule.
One unmatched segment — typically `cd` or an `echo` separator — forces a permission prompt even
when the rest is allow-listed, and one denied segment blocks the whole chain.

- Run one command per call; pass the working directory via the tool parameter, not `cd`.
- Don't use `echo`/`printf` as separators (deliberate: `Bash(echo:*)` would also allow
  `echo x > file`, bypassing file-edit rules).
- `Bash(cd:*)` is allow-listed as a fallback for the occasional chain that does contain `cd`.

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

**Audio burnout is multi-window and empirically calibrated, not a single fabricated snapshot.**
`Audio::process_audio_with_burnout` splits a recording into up to `max_windows` windows of
`window_seconds`, **evenly spaced across the whole file** (the old path scored only the first 10 s),
runs WavLM + features per window, rejects windows below `min_voice_activity_ratio`, and aggregates
per-field with `aggregation` (median/mean) before scoring. `Audio::add_burnout_analysis` then scores
against `audio::BurnoutConfig` (defaults in `BurnoutModels.h`, overridable via the optional
`burnout:` YAML section). The old fabricated baseline (`happy=0.30`, `pause_ratio=0.10`, ...) never
matched real WavLM output, so `positive_affect_loss` and `pause_tempo` saturated on every recording
and the score was near-constant (~0.5). Defaults are medians measured on `benchmark/control`. This
applies to the burnout path only — `audio::getDefaultBaseline()` (used by external influence) is
unchanged. Measure changes with `make build_harness` and `tools/burnout_harness.cpp`; there are no
labelled burnout positives, so only false-positive rate on the control sets is measurable.

**`src/common/httplib.h` is vendored but is not the request path.** Routing happens in the epoll
loop in `src/server/Server.cpp`. Don't add handlers to httplib expecting them to be reachable.

### Adding an API route requires up to three edits

1. Register in `post_routes_` or `get_routes_` — `Server.cpp:316-349`
2. Add to the `options_routes_` list for CORS preflight — `Server.cpp:352-358`
3. If parameterized (`/api/foo/<id>`), add prefix matching in the GET dispatcher — `Server.cpp:626-649`.
   Exact-match maps are checked first; prefix routes are hardcoded `if/else` on `path.find(...)`.

Missing step 2 or 3 fails silently as a 404 or a CORS error in the browser.

## Frontend (SPA)

- **The same binary serves the built SPA.** `make build_frontend` runs `npm install && npm run build`
  (Vite, `outDir: dist`, `assetsDir: static`); the server returns `frontend/dist/index.html` for any
  non-`/api` path, so client-side routes need no server config. `frontend/dist` is gitignored —
  source edits reach the running site only after a rebuild.
- **Routing is hand-rolled; `react-router-dom` is dead weight.** `src/context/NavigationContext.jsx`
  keeps `currentPage` and pushes history entries; `App.jsx` switches on it (`home`, `features`,
  `detector`, `privacy`, `contact`; unknown paths render `home`). react-router sits in
  `package.json` and the forced `react-vendor` chunk but is never imported in `src/`.
- **Two navigation helpers with different targets.** `navigateTo(page)` switches the top-level page
  (demo buttons → `detector`, every "Обсудить пилот" / header CTA → the `contact` page).
  `navigateToSection(id)` scrolls to an element of the landing page (double rAF + `scrollIntoView`);
  targets need both the `id` and CSS `scroll-margin-top: 92px` for the sticky header.
- **Landing anchor ids are the menu contract** (a rename fails silently — the menu just stops
  scrolling). In `Home.jsx`: `technology` (hero), `solutions` (analysis cards), `industries`,
  `cases` (benefit cards + decision banner), `demo` (CTA band), `contact` (CTA contact column);
  `about` is on the `<footer>` element in `Footer.jsx`.
- **Nothing sends email.** All "Обсудить пилот" buttons call `navigateTo('contact')`; `mailto:` /
  `tel:` links exist only as information on the contact page. A real email flow would need a new
  backend route — the binary has no SMTP code.
- **Assets:** `frontend/public/static/` referenced as `/static/<file>` (Vite copies `public/` into
  `dist/`). Branding is `razuma.svg` + `skolkovo.webp`; photos are `.webp`. The landing hero
  background lives in `landing.css` (`url('/static/hero.webp')`), not in JSX — swap the file there
  too when changing the image.
- **Translations:** `src/utils/translations.js` holds RU and EN; `t('a.b.c')` returns the key on a
  miss (a missing translation shows a raw key in the UI). Landing card lists are per-language arrays
  of objects paired with the `ANALYSIS_ICONS` / `BENEFIT_ICONS` arrays **by index** — reorder one,
  reorder the other.
- **Dead tooling:** `npm test` (jest) and `npm run lint` (eslint) both fail — no jest config, no
  eslint config, no test files. Only `npm run build` / `make build_frontend` work.

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
- **Frontend:** components are named exports in `src/components/`, one stylesheet per area in
  `src/styles/components/` (`landing.css` for the home page), tokens in `src/styles/global.css`
  (`--layout-padding-x`, `--color-*`). UI text goes through `t()` — add keys to both RU and EN
  halves of `translations.js`.

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
| `frontend/src/` | React 18 + Vite 5 SPA; landing = `components/Home.jsx` + `styles/components/landing.css`; i18n = `utils/translations.js` |
| `config/` | YAML profiles, `nginx` vhost template, `service` systemd unit |
| `docker/` | five compose variants, one per deployment topology |
| `training/` | PyTorch training and dataset prep |
| `tests/` | unit / integration / end_to_end / benchmark |

## Guardrails

- **Never commit** `config.yaml`, `models/`, `venv/`, `build/`, `uploads/`, `results/`, `logs/`,
  `data/`, `frontend/dist/` — all gitignored, and `config.yaml` holds credentials
  (`dragonfly.password`, `gigachat.auth_key`).
- **`contrib/emotiefflib` submodule is always dirty after `make install` — this is expected, not a
  problem.** The two modified files (`emotieffcpplib/CMakeLists.txt`,
  `models/prepare_models_for_emotieffcpplib.py`) are exactly the content of the tracked
  `emotiefflib.patch`, applied by `install_deps.sh:43`. The changes live in the parent repo as the
  patch file, so they are reproducible for anyone running `make install`. **Never commit inside the
  submodule or bump the parent's submodule pointer** — a local fork commit is unreachable for other
  clones and breaks their `git submodule update`, and `git apply` would then fail on re-install.
  The long-term fix is a PR upstream to `sb-ai-lab/EmotiEffLib`; after it merges, bump the pointer
  and delete the patch file plus the `git apply` line in `install_deps.sh`.
- `requirements.txt` is for **training/export only**. Adding a package there does not make it
  available to the running server.
- The systemd unit `config/service` will not start as written: `ExecStart` points at
  `build/EmotionAI` (actual binary is `build/emotionai`) and passes `--config`, which the binary
  does not parse.
- Default bind is `0.0.0.0:80`, which needs root or a `cap_net_bind_service` capability.
- A stray empty directory named `mkdir/` sits in the repo root — an artifact, not a real module.
