# Independent Code Review — Mobile branch (`emo-mobile-capacitor`)

**Scope:** the full diff of `emo-mobile-capacitor` against `git merge-base origin/main HEAD`
(`3a0d433..cfbbc56`, 146 files, +6967/−72). That is: the C++ security layer (app token,
rate limiter, device identity, resumable uploads, baseline storage), the SPA API client and
upload progress, the mobile shell and its Vite entry, the Capacitor Android/iOS projects and
assets, and the build/config/docs changes. I did **not** review the pre-existing inference
pipeline, MTCNN/audio code, the Compass SPA, or generated PNG bytes. No source code was
changed in this pass.

**Verdict:** the identity design is sound and the security-critical details are right —
constant-time token compare, SHA-256-only token storage, a clean app-token/device-token
header split with a legacy fallback, single-retry re-bootstrap, and a complete route
registration (POST map + the OPTIONS fallback). RU/EN translation parity for the new keys is
correct, and the `NavigationContext` change fixes a real deep-link bug on the web too. The
defects cluster in two places: **upload finalization performs heavy synchronous I/O on the
single-threaded event loop**, and **the deployment knobs are easy to misconfigure in the exact
nginx topology this repo ships**. None of the identity cryptography is wrong; most of the risk
is availability and silent-success, not impersonation.

---

## 🔴 High

### H1. Upload finalization copies the whole file on the epoll thread
`src/server/Server.cpp:2614` (`submitFileFromPath`), called synchronously from `:2441`
(`handleUploadComplete`); total dispatch is inline in `handleClientData` → `processRequest`.

`submitFileFromPath` runs `file_storage_->saveFileFromPath(...)` (a full copy to storage) and
then `fs::copy_file(local_source, upload_folder_/safe_filename)` (a second full copy) before it
enqueues the inference task. For a 300 MB video that is ~600 MB of blocking disk I/O on the
thread that serves every connection. The one-shot path had the same shape, but the resumable
API exists specifically to make multi-hundred-MB uploads normal, so this path turns a rare
stall into the expected case.

**Fix:** generate `task_id` first, return `202` immediately, and move the storage copy + local
copy into the existing thread pool (post an initial `task_status` so `/api/progress` shows
"finalizing upload"). Alternatively make the worker read directly from the temp part and copy
to storage from within the worker, which also removes the double copy.

### H2. The baseline endpoint reports success when the write failed
`src/emotionai/FileProcessor.cpp:417` (`save_user_baseline` returns `void`),
`src/server/Server.cpp` (`handleBurnoutBaseline` always returns `200 baseline_saved`).

If Dragonfly is unavailable or the `SETEX` fails, `save_user_baseline` only logs and returns;
the handler then tells the client the baseline was saved. Since device identity now makes the
baseline the durable per-device profile, this is silent data loss.

**Fix:** return `bool` from `save_user_baseline` and answer `503` (or `500`) on failure so the
client can retry.

### H3. The shipped nginx topology does not enable `trust_proxy_headers`
`config_template.yaml:154` (`trust_proxy_headers: false`),
`docker/docker-compose-all.yaml:118` (nginx-loadbalancer), `:150` (mounts `./config.yaml`).

The compose files mount `./config.yaml`, i.e. whatever the operator copied from
`config_template.yaml`, where `trust_proxy_headers` is **false**. Behind that nginx every
request reaches the app from the proxy socket, so `clientKey()` returns one address for all
users and the "per-client" limiter becomes a **global** 240 req/min cap (burst 80) for the
whole deployment. A demo with a handful of simultaneous users will start returning `429`.
(`config/config_all.yaml` does set `true`, but no compose file mounts it — it is only listed in
`README.md` as a profile.)

**Fix:** set `trust_proxy_headers: true` in the template used with nginx (or ship a
`config_nginx.yaml`), and/or detect a loopback peer + require an explicit
`trusted_proxy_cidrs` allow-list. At minimum, make the deployment doc say that a nginx
deployment **must** flip this flag.

---

## 🟠 Medium

### M1. Device-token minting has no global bound
`src/server/Server.cpp:2519` (`handleDeviceAuthCreate`).

Minting is limited to 10/min per IP (`RateLimiter(10, 3, 10000)`), but every token is stored
for 180 days and there is no global cap. With the default empty `security.api_token`, anyone
who finds the URL can mint identities; rotating source IPs (or simply waiting) grows Dragonfly
without bound. The limiter's own `max_keys` bounds its in-memory map, not the stored tokens.

**Fix:** add a global/total cap (e.g. a counter key with a daily quota), require the app token
for minting on public deployments, and/or shorten the TTL with a sliding refresh on use.

### M2. Startup/shutdown cleanup deletes *all* in-flight upload parts
`src/server/Server.cpp:2162` (`cleanupUploadSessions`).

The function removes every session temp file and then every regular file in
`<uploads>/.chunks`. A second instance sharing storage (the cluster topology) or a rolling
restart will delete another instance's in-flight parts, silently breaking resumable uploads
mid-transfer.

**Fix:** make the temp dir per instance (`<uploads>/.chunks/<instance_id>/`) or delete only
parts older than the session TTL instead of everything.

### M3. Chunk writes run under the sessions mutex on the event loop
`src/server/Server.cpp:2301` (`handleUploadChunk`).

The handler holds `upload_sessions_mutex_` while opening and writing up to ~8 MB to disk.
Because requests are served on the epoll thread, a slow disk stalls every connection, not just
this upload.

**Fix:** take a session snapshot under the lock, release it, write, then re-take the lock to
advance `received` (serialize per upload id). Or enqueue the write on the pool with a
per-upload ordering guarantee.

### M4. No automated coverage for the new security surface
`tests/unit/server/RateLimiterTest.cpp` is the only new test, and `libgtest-dev` is not
installed here so `make test` cannot run it.

There is no test at all for `DeviceAuth`, the auth gate, or the upload state machine — the
highest-risk code in the branch. The live checks I ran (bootstrap, isolation, revoke, 401/429,
409 resync) are not reproducible in CI.

**Fix:** add integration tests (they can use `tests/configs/integration_config.yaml`, already
opted out of device auth) for: bootstrap → baseline write/read isolation → revoke deletes the
baseline; 401 without/with a bad token; 429 + `Retry-After`; chunk offset `409` resync. Install
gtest in the dev image so the suite actually runs.

### M5. Device auth adds a Dragonfly round trip to every `/api/` request
`src/server/Server.cpp:2065` and the device-resolution block in `processRequest`; the token GET
is in `src/server/DeviceAuth.cpp` → `DragonflyManager::get_raw`.

Every `/api/` call with a token now does a Redis `GET`, including the Focus frame path at ~1
frame/s and the 1 s progress polling. This runs on the epoll thread; a slow Dragonfly stalls
all requests, and it is work duplicated on paths that do not need an identity (progress,
upload chunks, focus frames).

**Fix:** resolve the identity only on the endpoints that use it (baseline, revoke), or cache
`token-hash → user_id` for a few seconds in memory.

### M6. Enabling the app token breaks Prometheus scraping
`monitoring/prometheus.yml:17` scrapes `/api/metrics` with no credentials.

`/api/metrics` is not in `security.exempt_paths`, so setting `security.api_token` makes every
scrape `401`. The deployment goes blind exactly when it is hardened.

**Fix:** add `/api/metrics` to `exempt_paths` (and document the exposure) or configure a bearer
token in the scrape config.

### M7. Rate-limiter eviction is O(n) once the map is full
`src/server/RateLimiter.cpp:74` (`evictIfFullLocked`).

With `trust_proxy_headers: true`, an attacker rotating `X-Forwarded-For` forces a full scan of
up to `rate_limit_max_keys` (default 10 000) buckets per new key — a cheap request becomes a
10k-comparison operation. The map's memory is bounded, but the CPU cost is attacker-controlled.

**Fix:** evict approximately (random entry or a small LRU list) instead of scanning; or lower
`max_keys` for the public deployment.

---

## 🟡 Low / nit

### L1. Exempt paths are prefix-matched, not exact
`src/server/Server.cpp:2065` (`isAuthorized`). `path.rfind(exempt, 0) == 0` means `/api/health`
also exempts `/api/healthz`. Nothing collides today; a future `/api/health-admin` would silently
be public. Match exactly, or require a `/` boundary for prefix exemptions.

### L2. The device-issue limit is a hidden constant and survives `rate_limit_enabled: false`
`src/server/Server.cpp` (initialize: `RateLimiter(10, 3, 10000)`). It is not in the `security:`
section and is not disabled with the general limiter. Either make it configurable or document
the deliberate exception.

### L3. Baseline TTL is shorter than the device identity's
`src/emotionai/FileProcessor.cpp:417` uses `app.application_expiration` (30 d), while a device
token lives 180 d. A device that returns after 31 days has a valid identity and a `404` baseline.
Use the token TTL for baselines, or refresh the baseline TTL on read.

### L4. New backend error strings reach the UI untranslated
`frontend/src/api/client.js` (`errorFrom`) returns the server's `error` verbatim
(`device_token_required`, `invalid_device_token`, `Too many requests`, `Invalid file type`), and
`Detector` renders `t(error)`; `t()` returns the raw key on a miss. In RU mode these appear in
English. Add keys to both translation halves and map the known codes.

### L5. Malformed `size` yields `500`, not `400`
`src/server/Server.cpp:2188` (`handleUploadInit`). `json::value("size", -1LL)` throws
`type_error` for a string/number-float, which the outer catch turns into `500`. Check
`is_number_integer()` first and answer `400`.

### L6. A failed second copy orphans the storage object
`src/server/Server.cpp:2614`. `saveFileFromPath` writes to storage, then the local copy can
throw; the handler deletes the temp part but not the storage object. Delete the storage path on
failure, or copy locally first.

### L7. Unnecessary Android microphone permission
`mobile/android/app/src/main/AndroidManifest.xml:40`. The Focus camera requests
`getUserMedia({ audio: false })` and audio is chosen through the system file picker, so the app
never captures audio itself. `RECORD_AUDIO` is a dangerous permission and adds Play review
friction. Drop it unless in-app audio recording is planned.

### L8. iOS usage descriptions are Russian-only
`mobile/ios/App/App/Info.plist:26`. The app ships RU and EN but has no
`en.lproj/InfoPlist.strings`; an English-locale reviewer sees Russian. Add localized strings.

### L9. The app bundle ships the landing page's images
`frontend/vite.mobile.config.js:82` (`publicDir: frontend/public`) copies `hero.webp`,
`bank.webp`, `industry.webp`, … (~1.6 MB) into the app. The app only needs the privacy policy
and maybe the logo. Point `publicDir` at a small app-specific directory.

### L10. Hardcoded Linux JDK path in the committed Makefile
`Makefile:5` (`JAVA_HOME_ANDROID ?= /usr/lib/jvm/java-21-openjdk-amd64`). It is overridable, but
the default is wrong on macOS. Document the override or resolve a JDK 21 at runtime.

### L11. Dead API surface
`frontend/src/api/client.js` (`getBurnoutBaseline`, `saveBurnoutBaseline`) has no callers in
`src/`. Keep it if the next milestone wires a baseline UI; otherwise drop it so it does not
drift from the server contract it no longer exercises.

---

## What is genuinely good (do not "clean up")

- The app-token / device-token split is correct: `X-API-Key` for the shared gate,
  `Authorization` for identity, with a constant-time compare and a legacy fallback that
  explicitly avoids mistaking the shared token for a device token.
- Only `SHA-256(token)` is stored, and tokens are not enumerable. No raw bearer token exists in
  Dragonfly — verified live.
- The client bootstraps once, retries exactly once on `invalid_device_token` (no retry loop), and
  fails soft when identity is unavailable so anonymous endpoints keep working.
- The resumable protocol's `409 {received}` resume path is well chosen, and `saveFileFromPath`
  avoids re-reading the file into a `std::string` for local/NFS storage.
- Route registration is complete for the new endpoints (POST map; OPTIONS falls through to
  `handleOptions`), which is the repo's classic silent-404 trap.
- RU/EN translation parity for `mobile.*` and `detector.uploading` is correct.
- `NavigationContext` now initialises from the URL, fixing the `/focus` deep-link flash on the
  web as well as in the app.
- Test configs opt out of device auth, so the existing suites keep their old contract.

---

## Suggested order of action

1. **H1** — move upload finalization off the epoll thread (availability).
2. **H3** — set `trust_proxy_headers: true` for the shipped nginx topology (config only).
3. **H2** — propagate baseline write failures.
4. **M5**, **M1** — cache the identity lookup; bound token minting.
5. **M2**, **M3** — per-instance chunk dir; move chunk I/O off the loop.
6. **M4** — integration tests for auth/identity/upload; install gtest.
7. **M6** — decide on `/api/metrics` auth vs. an exempt path.
8. **M7**, then the Low items in file order.

---

## Remediation (same branch, after this review)

The three 🔴 High findings were fixed on `emo-mobile-capacitor`:

- **H1** — `submitFileFromPath` now writes an initial `task_status`, then enqueues the storage
  copy, the local copy and the processing on the thread pool; `handleUploadComplete` no longer
  deletes the temp part (the job owns it, and deletes it after the copies). Verified live:
  progress `20 → 100`, `.chunks` emptied after completion, storage file written.
- **H2** — `FileProcessor::save_user_baseline` returns `bool`; `handleBurnoutBaseline` answers
  `503 Baseline storage unavailable` when the write fails. Happy path re-verified (`200`).
- **H3** — `X-Real-IP` / `X-Forwarded-For` are now honored only when the socket peer matches
  `security.trusted_proxies` (default: loopback + RFC1918, covering nginx-on-host and the
  Docker bridge), and `trust_proxy_headers` defaults to `true`. Verified live: with a trusted
  peer, distinct `X-Real-IP` values get separate buckets (`200/200`); with an untrusted peer,
  the header is ignored and the second request is `429`.

Still open: **M1–M7** and **L1–L11**.
