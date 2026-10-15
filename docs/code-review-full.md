# Independent Code Review — Full Project

**Scope:** the entire EmotionAI codebase — the epoll HTTP server and routing, config,
DB/cluster/storage/metrics/MTCNN/AI subsystems, the image/video/audio/FileProcessor
orchestration layer, the Focus camera session, and the React SPA (excluding the burnout
scoring internals and burnout frontend, which were reviewed and fixed in a prior pass —
see the "Already fixed" note at the end).
**Verdict:** the hot path is solid and the architecture is coherent. The real defects
concentrate in the *second* layer of the system: a Dragonfly connection-pool counter that
leaks until the pool exhausts itself, a local-storage path-traversal check that is a
cosmetic substring test, a metrics pipeline that records fake 200s with unbounded label
cardinality, a cluster/distributed-task subsystem that is dead code and broken even if
compiled, a per-call Caffe model reload on the image hot path, and a blocking external
HTTP call on the event-loop thread. None of these block the single-node deployment the
project currently ships, but the pool leak and the storage traversal are real defects that
would bite in production as soon as the corresponding paths get traffic.

---

## 🔴 High

### H1. DragonflyDB connection-pool counter leaks — the pool exhausts itself
`src/db/DragonflyManager.cpp:97,117` (increments), `:153-161` (early return).

`get_connection()` increments `connection_pool_->in_use` when a connection is taken from
the pool or freshly created, then moves it into the per-thread `thread_connection_`. But
`return_connection()` hits the thread-local branch and returns **before** the `in_use--`
at the bottom of the function. So every distinct thread that ever touches the DB
permanently bumps `in_use` by one and never decrements it.

Why it matters: with `max_pool_size = 100`, after enough pool-worker / thread-pool threads
have each taken a connection once, the pool check fails, `get_connection()` waits 5 s and
returns `nullptr` — and every task-status write, batch read, and heartbeat on that machine
starts failing with "No DragonflyDB connection available." A long-running server under
normal load eventually degrades to total DB failure with no log line explaining why.

**Fix options:** (a) decrement `in_use` in the thread-local early-return branch (the counter
then only measures *pool-external* connections — the intended invariant); or (b) track
in-use connections per-`redisContext` in a small map and decrement on the connection's
destruction.

### H2. LocalFileStorage traversal check is cosmetic; `calculateFileHash` bypasses it
`src/storage/LocalFileStorage.cpp:17-44` (`resolvePath`), `:273-335` (`calculateFileHash`),
plus the per-call `file_mutex_`.

Three distinct defects in one class:

1. **Traversal.** `resolvePath` rejects any path containing the substring `".."` but never
   verifies the result stays under `base_path_`. A legitimate filename like `"report..pdf"`
   is wrongly rejected, while the guard is bypassed by (a) absolute paths —
   `fs::path(base) / "/etc/passwd"` *is* `/etc/passwd`, and `".."` is absent — and (b)
   symlink targets inside `base_path_` pointing outside it. The check is a string test, not
   a containment test.
2. **Bypass.** `calculateFileHash` opens `fs::path(base_path_) / file_path` directly,
   skipping `resolvePath` (and its `..` test) altogether, so an absolute `file_path`
   hashes any readable file on the box.
3. **Concurrency.** `file_mutex_` is a single member mutex held across the *entire*
   save/read/delete/copy/move, so all local-filesystem I/O in the process is serialized
   regardless of how many storage instances or files are involved.

Why it matters: the storage layer is the substrate for uploaded analysis files and results;
an absolute-path input is an arbitrary-file read, and the mutex turns the storage layer
into a process-wide throughput ceiling on uploads.

**Fix options:** normalize the joined path (`fs::canonical`/`weakly_canonical`) and assert
`path.starts_with(base_path_)`; use the same helper in `calculateFileHash`; drop the global
mutex or make it per-path.

### H3. Task-status writes go to a different key than reads when the pipeline falls back
`src/db/TaskManager.cpp:92` (pipeline), `:109` (fallback), `:176`, `:259`;
`src/db/DragonflyManager.cpp:374,405`.

The happy path is consistent: `BackgroundUpdater` writes raw keys `"task:" + task_id` via
`pipeline_set`, and reads build the same raw keys for `pipeline_get`. But when the pipeline
throws, the fallback calls `dragonfly_manager_->set_task_status(task_id, status)`, which
internally prepends **another** `"task:"` — writing to `task:task:<id>` — while every read
looks at `task:<id>`.

Why it matters: any transient Dragonfly hiccup that makes the pipeline fail silently splits
the key space. The in-memory cache still serves writes, so the divergence only surfaces
after cache expiry (5 min) or a restart: the task reports "not found" even though it
completed, and `batch_get_status` returns gaps for exactly the tasks that failed through the
fallback.

**Fix options:** give `DragonflyManager` a `set_task_status_raw` (no prefix) and use it in
the fallback so both paths share one key convention; or make all callers pass already-
prefixed keys and drop the implicit prefix in the manager.

---

## 🟠 Medium

### M1. `/api/metrics` data is wrong in three ways; the middleware layer is dead
`src/server/Server.cpp:718-735`; `src/metrics/MetricsCollector.cpp`; `src/metrics/MetricsMiddleware.cpp`;
`src/server/Server.h:203-211`.

1. **Fake status codes.** `processRequest` records `recordRequest(method, path, 200,
   duration)` on the success branch unconditionally — a handler that answered 404/405/429
   is recorded as a 200. Only the exception path records 500. The actual status sent to the
   client is never observed.
2. **Unbounded cardinality.** `recordRequest` keys maps by `method + ":" + path` with the
   *raw* path. Parameterized routes (`/api/results/<filename>`, `/api/progress/<task_id>`,
   `/api/focus/session/<id>`) create a new entry per unique ID forever — unbounded memory
   growth.
3. **Last-write-wins "gauges."** `request_durations_[key] = duration` overwrites instead of
   aggregating; the exported duration is the last observed value, not a histogram or
   average. Same for `task_durations_`.
4. **Dead layers.** `MetricsMiddleware` (the only thing whose purpose is wrapping these
   calls) has zero callers. `Server::updateRequestMetrics` is a dead wrapper; `Server`'s own
   counters (`total_requests_`, `active_connections_`, `endpoint_requests_`,
   `status_codes_`, `metrics_mutex_`) are declared, never incremented, never exported.

Why it matters: anyone dashboarding off `/api/metrics` gets an error-rate of ~0%, a
duration number that reflects the previous request, and a slowly leaking map. The dead
counters/middleware mislead anyone trying to extend the system.

**Fix options:** thread the actual status code out of each handler (or intercept at the
send choke points); normalize parameterized paths to their route template before keying;
replace the duration overwrite with sum/count; delete `MetricsMiddleware`,
`updateRequestMetrics`, and the five dead `Server` members.

### M2. Config knobs that are parsed and validated but never consumed
`src/config/Config.cpp:60-71,84,105-112,140,145-152,399-410`; `src/db/TaskManager.h:57-59`.

- `app.max_content_length` (60) — the request loop has no body-size cap. A client can
  declare a 10 GB `Content-Length` and the server will buffer it (unbounded). The one knob
  with real teeth (memory-exhaustion vector).
- `app.allowed_extensions` and `app.emotion_type` — never referenced outside `Config`.
- The entire `mtcnn` section — zero references to `config.mtcnn()` anywhere in `src/`.
- `task_management.{cache_ttl, max_cache_size, batch_size, batch_timeout_ms}` — `TaskManager`
  hardcodes `CACHE_TTL = 5 min`, `MAX_CACHE_SIZE = 10000`, batch cap 50.
- `queue.max_retries`, `queue.batch_size` — only `queue.poll_interval_ms` is used, and only
  inside the never-compiled `WITH_CLUSTER` block.
- `logging.flush_on_level` — parsed and validated, but `getSpdLogFlushLevel()` has no
  callers, so the flush policy is never applied to spdlog.

Why it matters: operators can set these in `config.yaml`, watch them echoed in the startup
log, and have zero effect — the classic "silently not honored" configuration trap.

**Fix options:** wire each knob into its consumer (content-length check in the read loop,
`TaskManager` cache constants from config, `flush_level` in `Logger::initialize`), or remove
the knobs and their parse/validate/log lines so the config file matches reality.

### M3. Cluster + distributed-task subsystem is dead code — and broken even if compiled
`src/cluster/`, `src/server/Server.cpp` (`#ifdef WITH_CLUSTER` blocks), `src/db/RedisManager.cpp`;
CMakeLists (no `WITH_CLUSTER` definition).

`WITH_CLUSTER` is referenced in `Server.cpp`/`Server.h` but is **never defined by any CMake
target** — the cluster manager, distributed workers, and `processDistributedTask` are not in
any current binary. The code that *would* ship if the flag were turned on is non-functional:

- `DistributedTaskManager::getNextTask` calls `get_task_status_json("queue:<name>:*")` —
  Redis `GET` takes a literal key, not a glob; the `*` is never expanded, so it returns nil
  for every queue and the worker loop idles forever.
- `submitTask` writes `task:queue:<name>:<id>`; `markTaskComplete` acquires a connection,
  does **nothing**, and returns it — the queue entry is never deleted. `markTaskFailed`
  writes the status under a *third* key.
- `ClusterManager::getClusterMembers` returns only the calling instance, so leadership
  election and `cleanupStaleInstances` operate on a one-member set — a 1-member cluster
  where every instance is its own leader.
- `acquireLock`/`releaseLock` are non-atomic GET-then-SET (two racers both acquire), and
  `releaseLock` never deletes the lock key.
- `RedisManager` (543 lines) is included by no one outside its own files — the runtime uses
  only `DragonflyManager`.

Why it matters: the largest dead-code surface in the repo ships behind a flag no build
sets, and its "TODO/simplified" comments would mislead the next person into thinking the
feature is nearly done.

**Fix options:** either finish it (real `LPOP`/`BRPOPLPUSH` queue ops, atomic `SET NX`
locks, `SCAN` for membership, key deletion in `releaseLock`/`stop`) or delete
`src/cluster/`, `RedisManager`, and the `#ifdef` blocks. Half-built infrastructure that
compiles to nothing is a liability.

### M4. Image error path returns a different JSON shape than the success path
`src/emotionai/FileProcessor.cpp:1009-1034`.

`process_image` has two return shapes for the same logical result:
- Success (delegated to `Image::process_image`): `{main_prediction: {index, label,
  probability}, additional_probs: {...}}`
- Error: `{emotion: "unknown"|"error", confidence: 0.0, additional_probs: {}, error: "..."}`

The frontend reads `result.main_prediction.label/probability` and `result.additional_probs`
— neither exists in the error shape. A failed image surfaces as a video frame silently
excluded from the average and as an image card that renders no emotion at all.

**Fix:** in the error branches, return the same `{main_prediction, additional_probs}` shape
with `main_prediction.label = "error"` / `"unknown"`, `probability = 0`, empty
`additional_probs`. Then the frontend's existing fallback and the video-averaging path both
work without changes.

### M5. `Image::recognizeFaces` reloads three Caffe networks on every call
`src/emotionai/Image.cpp:136-158`.

Each call builds a fresh `MTCNNDetector` from three `prototxt`+`caffemodel` pairs — three
Caffe model loads per call. This is invoked once per image upload, per video frame (up to
20× per video), and per Focus camera frame (~1 Hz). For a 30-second Focus session that is ~30
loads of the same weights; for a 20-frame video, 20 loads. The detector and its weights are
immutable per process, so this is pure repeated work on the hot path.

**Fix:** lift the `MTCNNDetector` into a `static` (or a member of `FileProcessor`, shared
under `model_mutex_`) and reuse it. Key the cache by the resolved model path if it can
change at runtime.

### M6. `resampleToModelRate` duplicates `Audio::resample_audio` and can drift
`src/emotionai/FileProcessor.cpp:427-449` is a verbatim copy of `Audio::resample_audio`
(`Audio.cpp:583-605`), with only the target rate hardcoded to 16000 instead of
`Audio::TARGET_SR`. Any future change to the interpolation algorithm or the target rate has
to be made in two places.

**Fix:** make `Audio::TARGET_SR` (and optionally `resample_audio` as a static) public and
call them from `resampleToModelRate`; or add a `static_assert`/test that
`EI_MODEL_SAMPLE_RATE == Audio::TARGET_SR`.

### M7. `process_video_file` frame-sampling reads every frame; dead constants
`src/emotionai/FileProcessor.cpp:873-904`.

1. The loop reads **every** frame from disk (`cap.read` for all frames) and then `continue`s
   on non-sampled frames. For a 30-minute 30 fps video that is 54 000 reads to extract 20
   frames — a real CPU/IO cost that scales with video length, not with the number of frames
   actually scored.
2. `MAX_VIDEO_FRAMES = 60` and `NUM_VIDEO_FRAMES = 15` (FileProcessor.h:95-96) are never
   referenced anywhere in `src/` — dead.
3. The forced last-frame line can produce more than the cap and trim the tail, so the
   reported `frames_processed` may not match the "evenly spaced" intent.

**Fix:** use `cap.set(cv::CAP_PROP_POS_FRAMES, frame_num)` to seek instead of sequential
read; delete the two dead constants; account for the forced last frame in the count.

### M8. `useFileUpload` emits translation keys that don't exist in either language
`frontend/src/hooks/useFileUpload.js:15,50,61` sets error state to the literal strings
`error_file_not_selected` and `error_upload_failed`. Neither key exists in
`frontend/src/utils/translations.js`, so `t()` returns the raw key, which is rendered
verbatim to the user via `Detector.jsx:137`. The sibling path in the same file
(`validators.js` → `errors.*` keys) is correct; these three literals are the outliers.

**Fix:** add `error_file_not_selected` and `error_upload_failed` to both the RU and EN
halves (or reuse the existing `errors.*` keys).

### M9. `handleFocusBreakdown` runs a blocking external HTTP call on the epoll thread
`Server.cpp:1562-1618` calls `ai_client_->breakdownTask(...)` synchronously inside the epoll
dispatch. `breakdownTask` is a blocking `curl_easy_perform` bounded only by
`CURLOPT_TIMEOUT` (default 30 s). A hung or slow model therefore stalls the *entire* server
— every other client's requests — for up to 30 seconds. Contrast with `handleFocusFrame`,
which correctly offloads inference to `thread_pool_` and returns 202 immediately.

**Fix:** offload `breakdownTask` to `thread_pool_` and return 202, or add a short connect
timeout and a non-blocking queue. The codebase already has the thread pool for exactly this.

---

## 🟡 Low / nits

**L1. S3 backend: unreachable reconnect, wrong `fileExists`, bucket auto-creation off the
config path.** `src/storage/S3FileStorage.cpp:132-156,260-278`. `reconnect` throws
"Reconnect not implemented" *before* any reconnect code — the function unconditionally
returns `false` and the client-rebuild below never runs, so a transient S3 outage has no
recovery path. `fileExists` returns `response && response.size > 0` — a legitimately
uploaded zero-byte object reports as non-existent. The live `createStorageFromConfig` S3
branch uses a 6-arg constructor that never sets `create_bucket_if_not_exists`, so bucket
auto-creation is unreachable from config.

**L2. AI prompt language framing.** `src/ai/AIClient.cpp:136-147`: the system prompt is
entirely Russian and only the *name* of the output language changes. For `lang=en` the model
is still framed as "Ты — ассистент Razuma… Ответь строго в формате JSON…" — a Russian system
prompt instructing English output. Make the system prompt language-neutral or per-language.

**L3. `FocusSession` resets timer and completion state on every plan-array identity change.**
`frontend/src/components/FocusSession.jsx:31-35` runs `useEffect(..., [plan, ...])` which
calls `onIndexChange(0)`, `onRunningChange(false)`, `onCompletedChange([])`. The dependency
is the *array identity* of `plan`, which is a fresh array on every `handleAddStep` /
`handleRemoveStep` / `handleApplySteps` in `Focus.jsx` — not only on full AI regeneration. A
mid-session "apply sub-steps" silently wipes `completedSteps` and resets the countdown.

**L4. `Privacy.jsx` references a translation key that doesn't exist.** `Privacy.jsx:24`
renders `t('privacy_error')`; the key is absent from both halves, so an iframe load error
shows the raw string. (Low exposure: `<iframe>` `onError` rarely fires in modern browsers.)

**L5. `axios` is a dead dependency; `frontend/src/api/endpoints.js` is a dead module.**
`axios ^1.6.0` is in `package.json` and bundled by `vite.config.js` but never imported in
`src/` (all HTTP goes through the `fetch`-based `apiClient`). `endpoints.js` exports
`ENDPOINTS` but is imported by nothing.

**L6. `useProgress` matches backend messages by exact English string; unmatched strings
leak raw.** `frontend/src/hooks/useProgress.js:24-40` maps server progress messages to
translation keys via exact/substring English comparisons. Any unmatched message (per-frame
progress, future wording drift) passes through as raw English to both RU and EN users.
`ProgressIndicator.jsx:14` compounds this by guarding translation with
`!text.includes(' ')`, so multi-word messages are never even attempted for `t()`.

**L7. `Image` file-backed constructor and helper methods are unused in the live path.**
`src/emotionai/Image.h:16-30`, `Image.cpp:12-21,40-70`. The `Image(const std::string&
filename)` constructor, `get_buffer`, `mime_bundle_repr`, `to_cv_mat`, and
`resizeWithAspectRatio` are never called from anywhere in `src/` — vestigial from an earlier
API where `Image` was a file-backed wrapper.

**L8. `process_video_file` comment says "Target 10–20 frames" but the actual cap is 20 and
the lower bound is 10.** The comment and the constants (`MIN_VIDEO_FRAMES = 10`,
`MAX_BATCH_VIDEO_FRAMES = 20`) are consistent with each other, but the dead
`MAX_VIDEO_FRAMES = 60` / `NUM_VIDEO_FRAMES = 15` constants (see M7.2) make the effective
sampling range ambiguous to a reader.

---

## ✅ What's genuinely good (call out so it isn't "fixed")

- **The hand-rolled epoll server is careful.** Bounded read buffer with `Connection: close`,
  header/body split on the exact 4-byte terminator, `Content-Length` parsed defensively,
  static-asset path checks, and a per-connection `ClientContext` that keeps the state
  machine explicit. The multipart parser handles realistic upload shapes cleanly.
- **Config lifecycle is disciplined.** `Config::loadFromFile` parses every section,
  `validate()` rejects bad values with specific error logs, and the startup log echoes the
  effective config — what-you-see is what-you-load. (The finding is about knobs that are
  *valid* but unwired, not about sloppy loading.)
- **`AIClient` does the unglamorous right things.** `extractJson` strips markdown fences
  before parsing; token usage is returned in the payload; a disabled model returns empty
  JSON and the caller degrades gracefully (502/503 + local template fallback per the EMO-19
  invariant); only task text + duration leave the process — no frames or personal data.
- **`TaskManager` caching design is sound in intent.** Two-level (in-process map +
  DragonflyDB) with a background batch writer that coalesces up-to-50 updates per 100 ms
  tick, LRU eviction, and hit/miss counters. (The key-mismatch finding is in the fallback,
  not the design.)
- **`ThreadPool` + in-memory `FocusSessionManager` keep the hot path lock-light.** Frames
  are scored on the pool, the session is a plain in-memory object with no disk or DB
  round-trips, and the whole session vanishes on `close` — matching the "nothing persisted"
  Focus contract.
- **The Focus session contract is correct.** `FocusSessionManager::classify` requires a face
  *and* both va_mtl heads to mark a frame valid, so the 7-class model never triggers an
  offer — matching the AGENTS.md spec exactly, with no special-casing.
- **`emotionPolicy.js` is a careful, well-commented port** of the v6 spec: robust
  `1.4826·MAD` z-scores with a 0.05 floor, duration-weighted persistence ratios, explicit
  NaN guards so "a missing channel is never filled with 0."
- **The external-influence path is well-bounded.** It slices the input to
  `max_fragments × fragment_seconds` *before* resampling, runs the quality gate before any
  WavLM work, and reports live progress per batch. A 1-hour silent call costs one quality
  estimate and no inference.
- **Landing anchor contract holds.** All `navigateToSection` targets resolve; every anchor
  id has `scroll-margin-top: 92px` for the sticky header. Icon index-pairing is correct
  (`ANALYSIS_ICONS` (3) and `BENEFIT_ICONS` (4) match the per-language card arrays).
- **RU/EN leaf-key parity is exact** — zero one-sided drift across all 1869 leaf keys per
  half (including `burnout.*`, audited in the prior pass).

---

## Suggested order of action

1. **H1** — fix the Dragonfly `in_use` leak. This is the only finding that degrades a
   *working* deployment into a broken one over time.
2. **H2** — fix the LocalFileStorage traversal check and the `calculateFileHash` bypass.
   Security-adjacent; low effort to make the check a real containment test.
3. **H3** — fix the task-status key mismatch in the pipeline-fallback path. Three lines.
4. **M9** — offload `handleFocusBreakdown` to the thread pool. One finding that stalls the
   whole server; the thread pool already exists.
5. **M5** — cache the MTCNN detector across calls. Direct perf win on the image/Focus
   hot path; small change.
6. **M2** — wire `app.max_content_length` into the request loop (the one knob with real
   teeth), and either wire or delete the rest.
7. **M1** — fix `/api/metrics` status codes, normalize parameterized paths, delete the dead
   middleware/counters.
8. **M4** — unify the image error-path JSON shape with the success shape.
9. **M3** — decide: finish the cluster subsystem or delete `src/cluster/`, `RedisManager`,
   and the `#ifdef WITH_CLUSTER` blocks.
10. **M7 / M6** — fix video frame-sampling (seek instead of read-all), delete dead
    constants, de-duplicate `resampleToModelRate`.
11. **M8** — add the missing `error_file_not_selected` / `error_upload_failed` translation
    keys.
12. **L1–L8** — batch the small fixes: S3 reconnect/fileExists, AI prompt language,
    FocusSession plan-reset, `privacy_error` key, dead `axios`/`endpoints.js`,
    `useProgress` string matching, dead `Image` methods.

---

## Already fixed (prior burnout review pass — do not re-flag)

The following burnout-subsystem findings were identified and fixed in a prior review pass
(this report deliberately does not re-raise them):

- **H1 (burnout):** dead history-aware state branch removed; single-recording state bands
  documented; `BURNOUT_LIKE` unreachable by design.
- **H2 (burnout):** confidence made data-driven from real recording duration, scored-window
  count, and baseline reliability (weights 0.40/0.20/0.40).
- **M1 (burnout):** two disagreeing baselines collapsed to one source of truth
  (`BurnoutConfig::defaultBaselineJson()`); stale `getDefaultBaseline()` deleted;
  `ExternalInfluenceAnalyzer` now uses the calibrated baseline.
- **M3/M4 (burnout):** component-key map unified into `constants.js`; `BurnoutKeyComponent`
  and `burnoutPdf.js` both use the shared `BURNOUT_COMPONENT_KEYS`.
- **L2 (burnout):** `burnout.conclusionText.*` / `burnout.urgencyText.*` keys verified to
  exist in both RU and EN halves.
- **Bonus:** `analyze()` zeroes `risk`, `score`, `confidence` on every `INSUFFICIENT_DATA`
  early-return via a `refuse()` helper.
