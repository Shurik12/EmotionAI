# Razuma Focus — Architecture

Two product halves:

- **AI Task Breakdown** — user types a task, an OpenAI-compatible model splits it into 3 actionable steps.
- **Emotional Dynamics** — a consented camera stream feeds per-frame face + emotion inference into a server-side in-memory session; the browser runs the `emotion-pilot-v1` decision policy against a user-confirmed baseline and shows "Numa" assistant soft-offer cards.

---

## Data Flow

```
Browser                                    C++ binary
───────                                    ──────────
Focus.jsx (plan state)                     Server (epoll, src/server/Server.cpp)
 │ generate() → POST /api/focus/breakdown   │
 │                                         ├─ handleFocusBreakdown (inline)
 │ (local fallback taskBreakdown.js)       │      └─ AIClient (src/ai) → LLM
 │                                         │
FocusCamera.jsx                            ├─ POST /api/focus/session        → FocusSessionManager.createSession
 │ captureBlob → POST /frame (multipart)   ├─ POST /api/focus/session/<id>/frame
 │ poll  GET  /api/focus/session/<id>      │      └─ thread_pool → FileProcessor.process_image_frame
 │ decideEmotionOffer (emotionPolicy.js)   │             └─ Image::process_image (MTCNN + ENet)
 │ Numa.jsx gestures, focusSound.js        │      └─ FocusSessionManager.addResult (classify/normalize)
 │ FocusSession.jsx timer/steps            └─ GET/POST .../close
```

---

## 1. Frontend

### `frontend/src/components/Focus.jsx` — page container

- Three tabs: `movement` (planning), `session`, `how`.
- Owns **all shared session state** — `plan`, `stepIndex`, `sessionRunning`, `completedSteps` — so the camera card can point at / pause the same session.
- `generate(text)`: calls `apiClient.breakdownFocusTask(text, duration, language)`; on **any** failure (502/503/network) falls back to `breakdownTask(text)` local templates and shows a fallback notice.
- **Parking detail:** the session panel (timer + camera) stays **mounted** on other tabs — wrapped in `div.focus-panel-session.is-parked` (CSS `visibility:hidden`, not unmounted) — so the camera stream and baseline survive tab switches.

### `frontend/src/components/FocusCamera.jsx` — the core of the emotional-dynamics half

- **Consent-gated** camera: nothing starts until a checkbox is ticked; `getUserMedia({video:{ideal 640×480}})`; never records.
- **Session lifecycle:** `startSession()` → `apiClient.createFocusSession()`; `stopCamera()` → `apiClient.closeFocusSession(id)`; unmount cleanup always calls `stopCamera()`.
- **Frame pipeline:** `captureBlob()` draws the current `<video>` frame to a canvas → JPEG blob (quality 0.85). Auto sampling every 1 s or 2 s (the policy needs ≥8 answers inside a 15 s window, slower rates can never satisfy it) plus a manual "check frame" button.
- **Polling:** every 1000 ms `GET /api/focus/session/<id>` → `samples` state (last 60 frames kept for the 15 s window; 40 shown in the dynamics bar strip).
- **Policy execution (runs in the browser):** an effect re-runs `decideEmotionOffer(observation, context)` on every frame/baseline/context change. `context` carries `visible: !document.hidden`, `running`, `hasCurrentStep`, `enabled: offersEnabled` (**opt-in, default OFF**), `stepId`, and offer memory.
- **Card state machine** (15 spec scenarios): kinds `question`, `current-action`, `help`, `steps`, `pause`, `done`. Answer buttons map to callbacks (`continue`, `pause`, `help-return`, `show-action`, `split-steps`, `apply-steps`, `back`, `resume-session`).
- **Baseline confirmation UI:** `baselineReady()` gates the "confirm baseline" button (≥10 s, ≥8 frames, ≥80% coverage); confirmed baseline is a frozen snapshot for the session.
- **Numa mood:** open card wins; otherwise `ambientNumaMood(frames)` follows the dominant emotion over last 8 usable frames.

### `frontend/src/components/FocusSession.jsx` — work session timer

- Durations 5/15/25 min; per-step countdown; start/pause/resume, next-step, select step, reset; progress bar.

### `frontend/src/components/Numa.jsx` — the assistant character

- 2D cutout rig from layer PNGs, composited with CSS transforms.
- Gestures: `still, check, help, support, point, pause, done`.
- Maps page moods (`idle, rest, focus, thinking, help, check, support, point, happy, done, pause`) to gestures.
- Plays once per `reaction` bump (rAF animation), respects `prefers-reduced-motion`.

### `frontend/src/utils/emotionPolicy.js` — the decision engine (see §5)

### `frontend/src/utils/taskBreakdown.js` — offline fallback

- `detectTaskType(text)` keyword-matches (RU+EN) → `presentation | letter | study | generic`.
- `breakdownTask()` returns an array of **translation keys** resolved via `t()`.

### `frontend/src/utils/focusSound.js`

- Web Audio synthesis of two spec timbres (no audio files): `check` = 440 Hz sine, `done` = 523+659 Hz.

### API client

- `frontend/src/api/client.js`: `createFocusSession()`, `getFocusSession(id)`, `sendFocusFrame(id, blob)`, `closeFocusSession(id)`, `breakdownFocusTask(text, duration, lang)`.

---

## 2. Backend Session Manager — `src/focus/`

### `FocusSessionManager.h` / `.cpp`

- **Singleton**, in-memory `std::map<session_id, Session>` under a `std::mutex`. Nothing persisted; sessions are dropped on close.
- `Session` = `{samples: vector<json>, total: size_t, created_ms, updated_ms}`; max 1800 samples (≈30 min at 1 Hz) with oldest-first eviction.
- `createSession()` — UUID via `DragonflyManager::generate_uuid()`.
- `addResult(id, result)` — runs `classify(result)`, stamps index + timestamp, appends, evicts overflow.
- `getSession(id)` — returns `{session_id, count, total, samples, dynamics:{count, valid_count, mean_arousal, mean_intensity}}` or null (→ 404).

### `classify(result)` — the adapter to the normalized `EmotionFrame`

```
{valid, valence, arousal, intensity, emotions:{name:score}, label?, probability?}
```

- Reads `additional_probs` from the core result; accepts decimal strings or numbers.
- Category mapping: `happiness` → `joy`; whitelist of known categories; unknown keys skipped.
- **`valid` requires a face AND both valence and arousal heads present.**
  - With the 7-class model (`enet_b2_7.pt`) there are no regression heads → every frame stays `valid:false` → offers are structurally impossible.
- Valence clamped to [−1,1]; arousal rescaled to [0,1].
- Emits **data only, never human-readable text**.

### Where frames get scored

`FileProcessor::process_image_frame(bytes)` → `cv::imdecode` → `Image::process_image` (MTCNN face detection + ENet emotion model) → raw result. **No disk write, no task record** — in-memory only.

---

## 3. Server Routes — `src/server/Server.cpp`

| Method | Path | Handler | Response |
|--------|------|---------|----------|
| POST | `/api/focus/session` | `handleFocusSessionCreate` | `201 {"session_id": ...}` |
| POST | `/api/focus/session/<id>/frame` | `handleFocusFrame` | `202 {"session_id", "queued": true}` |
| POST | `/api/focus/session/<id>/close` | `handleFocusSessionClose` | `200 {"closed": true}` |
| GET | `/api/focus/session/<id>` | `handleFocusSessionGet` | `200 {session data}` or `404` |
| POST | `/api/focus/breakdown` | `handleFocusBreakdown` | `200 {"steps":[...]}` / `502` / `503` |

- Frame scoring is **queued on the thread pool** — the epoll thread stays non-blocking, returns 202 immediately.
- Breakdown is **synchronous** (inline in the epoll handler; the model is a fast flash variant).

---

## 4. AI Client — `src/ai/`

- `AIClient::breakdownTask(text, duration, lang)` → `{"steps":[...]}` or null.
- Plain **libcurl** POST to `{base_url}/chat/completions`, Bearer auth, temperature 0.4.
- Fixed Russian system prompt asking for exactly 3 JSON steps.
- `extractJson()` pulls the outermost `{…}` from raw model text (handles code fences/prose).
- **Privacy invariant:** only task text + block duration ever leave the process. Camera frames, emotions, personal data are never sent to the model.

---

## 5. Emotion Policy — `frontend/src/utils/emotionPolicy.js`

Faithful port of the v6 spec (`Razuma_Focus_Developer_v6_Emotion_Rules`, profile `emotion-pilot-v1`). Every threshold is a **pilot hypothesis, not a validated attention classifier**.

### Key constants

| Parameter | Value |
|-----------|-------|
| Baseline duration | ≥ 10 s |
| Baseline frames | ≥ 8 |
| Baseline coverage | ≥ 80% |
| Decision window | 15 s |
| Persistence threshold | ≥ 80% |
| Z-deviation trigger | 2σ |
| Scale floor | 0.05 |
| Max inter-frame gap | 5 s |
| Offer cooldown | 300 s (5 min) |
| Max offers per session | 2 |
| Offer lifetime | 20 s |

### Decision pipeline

1. **Guard cascade** — `outside-session`, `no-step`, `not-working`, `preferences-off`, `adapter-unverified`, `insufficient-signal`, `repetition-limit`, `baseline-required` → `quiet`.
2. **Signal checks** — ≥8 recent frames, fresh within 5 s, window coverage.
3. **Four patterns** (each requiring ≥80% persistence over its own window):

| Pattern | Condition | Numa mood | Primary action |
|---------|-----------|-----------|----------------|
| `check` | Flat valence, no joy/surprise, low arousal & intensity | check | continue |
| `fear-high` | Negative valence, z(valence)≤−2, z(arousal)≥2, z(intensity)≥2, fear dominates | support | continue |
| `friction-high` | Same cluster + anger/disgust ≥ 2σ | help | split-steps |
| `sadness-low` | Negative valence, low arousal, sadness ≥ 2σ | point | show-action |

### Robust statistics

- Per-channel z-score: `center = median`, `scale = max(1.4826·MAD, 0.05)`.
- Channel trusted only with ≥8 finite values covering ≥80% of valid baseline frames.
- Untrusted channels return NaN → rules fail cleanly (never fabricate signal).

---

## 6. Configuration

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| `ai:` | `enabled` | `false` | Must be true + `api_key` set for breakdown |
| `ai:` | `base_url` | `https://shared1.multitool.works:4000/v1` | OpenAI-compatible endpoint |
| `ai:` | `model` | `deepseek-v4-flash` | |
| `ai:` | `api_key` | *(empty)* | Gitignored in `config.yaml` |
| `ai:` | `timeout_seconds` | `30` | |
| `model:` | `emotion_model_path` | `models/enet_b0_8_va_mtl.pt` | Must be 8-class for policy to fire |

No other Focus-specific config. Camera sampling rate, policy thresholds, offer lifetime are hardcoded in the frontend.

---

## 7. Tests & Benchmarks

| What | Where | Notes |
|------|-------|-------|
| Focus HTTP benchmarks | `tests/benchmark/FocusBenchmark*` | Google Benchmark, runs against a live server |
| LLM token-cost benchmark | `tests/benchmark/AiCostBenchmark*` | Sends 10 sample tasks to `/api/focus/breakdown`, records tokens |
| Benchmark runners | `make benchmark_focus`, `make benchmark_ai_cost` | Save JSON to `tests/benchmark/measurements/` |
| Baseline measurement | `tests/benchmark/measurements/focus-20261014-205410.json` | Frame upload: 389–520 RPS; full flow: 110–161 RPS |
| Unit tests | *none yet* | No `tests/unit/*Focus*` |

**RPS requirement:** > 10 RPS. **Measured:** ~400–500 RPS on frame upload (1–32 concurrent clients). Met with 40× margin.

**LLM cost tracking:** Every `/api/focus/breakdown` call records token usage
(`prompt_tokens`, `completion_tokens`, `total_tokens`) in:
- The HTTP response body (`usage` field)
- Prometheus counters (`ai_calls_total`, `ai_prompt_tokens_total`,
  `ai_completion_tokens_total`, `ai_total_tokens_total`) via `/metrics`
- The offline benchmark (`make benchmark_ai_cost` → `measurements/ai-cost-*.json`)

**Cost per DAU** = `ai_total_tokens_total (24 h) / DAU × price_per_token`.

---

## Key Invariants

1. **Privacy:** Camera frames go only into the in-memory session — never persisted to disk. The LLM receives only task text and block duration.
2. **The backend is only an adapter.** All decision logic lives in the browser; the server normalizes scale and emits data, never text.
3. **Missing data is never fabricated.** `valid` requires a face + both VA heads; the policy fails on NaN instead of zero-filling; 7-class model ⇒ no offers ever.
4. **Opt-in and rate-limited:** Offers default off, one per step, ≥5 min apart, ≤2 per session, self-dismissing after 20 s. Camera only after explicit consent. Everything vanishes on session close.
5. **Honesty framing:** Every threshold is documented as a pilot hypothesis — "an invitation to a conversation, not a concentration assessment."
