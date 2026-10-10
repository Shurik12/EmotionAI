// Razuma Focus emotion policy — improved 16-scenario version (EMO-17).
//
// This is the JavaScript port of the intervention spec in
// `docs/focus/FOCUS_INTERVENTION_SPEC.md` (source: ORI animation prototypes
// in `focus-assets/`). It replaces the earlier 3-pattern
// `emotion-pilot-v1` profile with four automatic offer rows (1–4) plus the
// passive states (5–8) and the user-driven/edge scenarios (9–16) that the
// component layer implements.
//
// The policy is a pure function over a normalized observation:
//   - `EmotionFrame` = { timestampMs, valid, valence, arousal, intensity, emotions }
//     with valence in [-1, 1] and arousal / intensity / category scores in [0, 1].
//   - `EmotionObservation` = { source, normalizedScaleVerified, baseline:{confirmedOnTask,frames}, frames }.
// The EmotionAI adapter (src/focus/FocusSessionManager) produces the normalized
// frames from the core response; this module owns the decision logic.
//
// Notation (see the spec): `z` is the change relative to the user's confirmed
// working baseline (robust z-score, `1.4826 · MAD` with a 0.05 floor), `V`
// valence, `A` activation, `I` expression intensity. `z >= 2` is a rise and
// `z <= -2` a drop. Within a row all conditions are ANDed unless marked OR,
// must hold for at least `patternPersistence` of the window (pilot: 60%, softer
// than the spec's nominal 80% because per-frame estimates are noisy) and
// persist in the latest valid frame. To tolerate live per-frame noise the
// z-scores are median-smoothed over `smoothFrames` and the window is scored by
// the fraction of usable frames that match (not by requiring both endpoints of
// each interval), so a single bad frame cannot cancel an otherwise sustained
// pattern. A missing channel disables any rule that depends on it — it is never
// substituted with zero (scenario 16).
//
// Every threshold is a pilot hypothesis, not a validated attention classifier.

export const emotionPolicy = Object.freeze({
  version: 'emotion-pilot-v2',
  baselineMs: 60000, // minimum confirmed working baseline
  windowMs: 15000, // default decision window (rows 2–4)
  checkWindowMs: 30000, // row 1 uses a longer window
  persistence: 0.8, // 80% temporal persistence (channel coverage)
  baselineCoverage: 0.25, // pilot: real webcams drop frames (no face / motion),
  // so a strict 80% coverage of the wall-clock window is unreachable. Readiness
  // is judged on enough valid frames spanning ≥60 s at ≥25% coverage.
  patternPersistence: 0.6, // pilot: per-frame emotion estimates are noisy, so
  // the offer window uses a softer threshold than the spec's nominal 80%.
  // Keep the latest-frame requirement; tune against logged `progress` values.
  smoothFrames: 3, // median-of-window smoothing of the z-scores; a single
  // noisy frame must not break a rule (pilot noise tolerance).
  deviation: 2, // z-score deviation threshold
  scaleFloor: 0.05, // MAD floor
  maxGapMs: 5000,
  maxStaleMs: 10000, // pilot: the decision window is anchored to the last valid
  // frame; if no face has been seen for longer than this, stay quiet. Detection
  // is sparse (~1 valid frame per 3 s), so this is looser than maxGapMs.
  cooldownMs: 300000, // one automatic offer per 5 minutes
  maxOffersPerSession: 2,
  offerLifetimeMs: 20000, // scenario 15: card closes after 20 s
  minFrames: 8, // ≥8 answers in the window
  minBaselineFrames: 16,
});

// Automatic offers (rows 1–4). `pattern` is the stable id used by the UI and
// the translation keys; `scenario` is the spec row number. `gesture` names the
// Numa animation and `sound` the optional cue (only rows 1 and 14 have sound).
export const emotionPatterns = Object.freeze({
  check: { scenario: 1, mood: 'check', gesture: 'check', sound: 'tu', windowMs: 30000 },
  help: { scenario: 2, mood: 'help', gesture: 'help', sound: null, windowMs: 15000 },
  support: { scenario: 3, mood: 'support', gesture: 'support', sound: null, windowMs: 15000 },
  point: { scenario: 4, mood: 'point', gesture: 'point', sound: null, windowMs: 15000 },
});

export const emptyOfferMemory = () => ({ lastOfferAt: null, offeredStepIds: [], count: 0 });

// Categories referenced by the policy. The spec lists nine, but the shipped
// core model (`enet_b0_8_va_mtl.pt`) exposes only eight — `anger`, `contempt`,
// `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`. `interest` and
// `shame` are therefore treated as optional in row 1 (see `buildRules`): when
// absent the clause is skipped and the row degrades to the spec's
// "experimental check" instead of being disabled outright.

const median = (values) => {
  const a = [...values].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
};

const finiteOr = (value) => (Number.isFinite(value) ? value : null);
const maxFinite = (...values) => {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.max(...finite) : -Infinity;
};

function usable(f) {
  return (
    f.valid === true &&
    [f.timestampMs, f.valence, f.arousal, f.intensity].every(Number.isFinite) &&
    f.valence >= -1 &&
    f.valence <= 1 &&
    f.arousal >= 0 &&
    f.arousal <= 1 &&
    f.intensity >= 0 &&
    f.intensity <= 1 &&
    !!f.emotions &&
    Object.values(f.emotions).every(
      (x) => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 1,
    )
  );
}

function ordered(frames) {
  return frames.every(
    (f, i) =>
      Number.isFinite(f.timestampMs) &&
      (i === 0 ||
        (f.timestampMs > frames[i - 1].timestampMs &&
          f.timestampMs - frames[i - 1].timestampMs <= emotionPolicy.maxGapMs)),
  );
}

// Fraction of the window's usable frames that satisfy the predicate. A rule does
// not need both endpoints of an interval to match, so independent per-frame
// noise does not square the miss rate. Used for the offer windows.
function frameRatio(frames, predicate) {
  let total = 0;
  let pass = 0;
  for (const f of frames) {
    if (!usable(f)) continue;
    total += 1;
    if (predicate(f)) pass += 1;
  }
  return total ? pass / total : 0;
}

// Fraction of the trailing baseline run made of usable frames. Real webcams
// drop frames (no detected face / motion), so this is the honest readiness
// signal — and it matches the "valid frames" count the UI shows.
export function baselineCoverage(frames) {
  if (!Array.isArray(frames) || frames.length === 0) return 0;
  return frames.filter(usable).length / frames.length;
}

// Mirrors the baseline checks in decideEmotionOffer so the UI can enable
// "confirm baseline" only when it will actually be accepted.
export function baselineReady(frames) {
  if (!Array.isArray(frames) || frames.length === 0 || !ordered(frames)) {
    return false;
  }
  const valid = frames.filter(usable).length;
  if (valid < emotionPolicy.minBaselineFrames) return false;
  const start = frames[0].timestampMs;
  const end = frames[frames.length - 1].timestampMs;
  return (
    end - start >= emotionPolicy.baselineMs &&
    baselineCoverage(frames) >= emotionPolicy.baselineCoverage
  );
}

// Build a robust z-score reader over the confirmed baseline. Returns NaN for a
// channel that is not covered by the baseline, which disables dependent rules.
function makeDeviation(baselineFrames) {
  const validBaseline = baselineFrames.filter(usable);
  const read = (f, k) =>
    k === 'valence' || k === 'arousal' || k === 'intensity' ? f[k] : f.emotions[k];
  const scales = new Map();
  return (f, k) => {
    const value = read(f, k);
    if (value === undefined) return NaN;
    let s = scales.get(k);
    if (!s) {
      const a = validBaseline.map((x) => read(x, k)).filter((x) => x !== undefined);
      if (a.length < emotionPolicy.minBaselineFrames || a.length / validBaseline.length < emotionPolicy.persistence) {
        return NaN;
      }
      const center = median(a);
      s = {
        center,
        scale: Math.max(
          1.4826 * median(a.map((x) => Math.abs(x - center))),
          emotionPolicy.scaleFloor,
        ),
      };
      scales.set(k, s);
    }
    if (!Number.isFinite(value)) return NaN;
    return (value - s.center) / s.scale;
  };
}

// Median-of-window smoothing of the z-scores (pilot noise tolerance). A single
// noisy frame should not break a rule, so each frame reads the median z of the
// last `windowSize` frames. Because z is a linear transform of the raw value,
// the median of z equals the transform of the median raw value, so this is the
// same as smoothing the signal before standardizing. A channel with no baseline
// scale stays NaN (missing channels never get filled with zero, scenario 16).
function makeSmoothedReader(zRaw, frames, windowSize) {
  const index = new Map();
  frames.forEach((f, i) => index.set(f, i));
  const cache = new Map();
  const series = (k) => {
    let s = cache.get(k);
    if (!s) {
      // Only usable frames carry a real measurement; an invalid (no-face) frame
      // still has numeric zeros, so it must not enter the median.
      const raw = frames.map((f) => (usable(f) ? finiteOr(zRaw(f, k)) : null));
      s = raw.map((_, i) => {
        const values = [];
        for (let j = Math.max(0, i - windowSize + 1); j <= i; j += 1) {
          if (Number.isFinite(raw[j])) values.push(raw[j]);
        }
        return values.length ? median(values) : NaN;
      });
      cache.set(k, s);
    }
    return s;
  };
  return (f, k) => {
    const i = index.get(f);
    return i === undefined ? NaN : series(k)[i];
  };
}

// All channels a row depends on must be present (and have a baseline scale).
const ready = (z, f, keys) => keys.every((k) => Number.isFinite(finiteOr(z(f, k))));

// Row definitions. Order matters: rows 2 and 3 are mutually exclusive on ties
// (`>` vs `>=`), row 1 and row 4 cannot co-occur with rows 2–3.
function buildRules(z) {
  const D = emotionPolicy.deviation;
  const zV = (f) => z(f, 'valence');
  const zA = (f) => z(f, 'arousal');
  const zI = (f) => z(f, 'intensity');
  return [
    {
      pattern: 'check',
      scenario: 1,
      windowMs: emotionPolicy.checkWindowMs,
      // Row 1's mandatory channels. `interest` / `shame` are optional: the
      // shipped eight-class core has no scale for them, so their clauses are
      // skipped and the row becomes the spec's emotional-only experimental
      // check rather than being permanently disabled.
      keys: [
        'joy',
        'surprise',
        'fear',
        'anger',
        'sadness',
        'contempt',
        'disgust',
        'valence',
        'arousal',
        'intensity',
      ],
      test: (f) => {
        const interest = z(f, 'interest');
        const neg = maxFinite(
          z(f, 'fear'),
          z(f, 'anger'),
          z(f, 'sadness'),
          z(f, 'shame'),
          z(f, 'contempt'),
          z(f, 'disgust'),
        );
        return (
          (!Number.isFinite(interest) || interest <= -D) &&
          z(f, 'joy') <= 0 &&
          z(f, 'surprise') <= 0 &&
          neg < D &&
          Math.abs(zV(f)) < D &&
          zA(f) <= -D &&
          zI(f) <= -D
        );
      },
    },
    {
      pattern: 'help',
      scenario: 2,
      windowMs: emotionPolicy.windowMs,
      keys: ['anger', 'disgust', 'fear', 'valence', 'arousal', 'intensity'],
      test: (f) => {
        const rise = maxFinite(z(f, 'anger'), z(f, 'disgust'));
        return (
          rise >= D &&
          rise > z(f, 'fear') &&
          f.valence < 0 &&
          zV(f) <= -D &&
          zA(f) >= D &&
          zI(f) >= D
        );
      },
    },
    {
      pattern: 'support',
      scenario: 3,
      windowMs: emotionPolicy.windowMs,
      keys: ['fear', 'anger', 'disgust', 'valence', 'arousal', 'intensity'],
      test: (f) => {
        const fear = z(f, 'fear');
        return (
          fear >= D &&
          fear >= maxFinite(z(f, 'anger'), z(f, 'disgust')) &&
          f.valence < 0 &&
          zV(f) <= -D &&
          zA(f) >= D &&
          zI(f) >= D
        );
      },
    },
    {
      pattern: 'point',
      scenario: 4,
      windowMs: emotionPolicy.windowMs,
      // Row 4 does not use intensity.
      keys: ['sadness', 'valence', 'arousal'],
      test: (f) =>
        z(f, 'sadness') >= D && f.valence < 0 && zV(f) <= -D && zA(f) <= -D,
    },
  ];
}

// Passive states (rows 5–8) never produce a reaction. They exist only so the UI
// can explain why Numa stays still; they must not be shown as an offer.
function passiveState(z, f) {
  if (!f || !usable(f)) return 'working';
  const D = emotionPolicy.deviation;
  if (finiteOr(z(f, 'joy')) >= D || finiteOr(z(f, 'interest')) >= D) return 'joy-interest';
  if (finiteOr(z(f, 'surprise')) >= D) return 'surprise';
  if (finiteOr(z(f, 'shame')) >= D || finiteOr(z(f, 'contempt')) >= D) return 'shame-contempt';
  return 'working';
}

export function decideEmotionOffer(observation, context) {
  if (!context.session || !context.visible) return { action: 'quiet', reason: 'outside-session' };
  if (!context.hasCurrentStep) return { action: 'quiet', reason: 'no-step' };
  if (!context.running || context.supportOpen) return { action: 'quiet', reason: 'not-working' };
  if (context.enabled !== true) return { action: 'quiet', reason: 'preferences-off' };
  if (
    !['synthetic-demo', 'razuma-core'].includes(observation.source) ||
    observation.normalizedScaleVerified !== true
  ) {
    return { action: 'quiet', reason: 'adapter-unverified' };
  }
  if (!Number.isFinite(context.nowMs) || !context.stepId || !context.memory) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }

  const memory = context.memory;
  if (
    memory.count >= emotionPolicy.maxOffersPerSession ||
    memory.offeredStepIds.includes(context.stepId) ||
    (memory.lastOfferAt !== null && context.nowMs - memory.lastOfferAt < emotionPolicy.cooldownMs)
  ) {
    return { action: 'quiet', reason: 'repetition-limit' };
  }

  const baseline = observation.baseline;
  if (
    !baseline?.confirmedOnTask ||
    !Array.isArray(baseline.frames) ||
    baseline.frames.length < emotionPolicy.minBaselineFrames ||
    !ordered(baseline.frames)
  ) {
    return { action: 'quiet', reason: 'baseline-required' };
  }
  const bs = baseline.frames[0].timestampMs;
  const be = baseline.frames[baseline.frames.length - 1].timestampMs;
  if (
    be - bs < emotionPolicy.baselineMs ||
    baselineCoverage(baseline.frames) < emotionPolicy.baselineCoverage
  ) {
    return { action: 'quiet', reason: 'baseline-required' };
  }
  if (
    !Array.isArray(observation.frames) ||
    observation.frames.length < emotionPolicy.minFrames ||
    !ordered(observation.frames)
  ) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }

  const z = makeSmoothedReader(
    makeDeviation(baseline.frames),
    observation.frames,
    emotionPolicy.smoothFrames,
  );
  // The spec requires the pattern to persist in the last VALID measurement, not
  // the literal last frame — with a webcam most frames can have no detected
  // face. Anchor the decision window to the most recent usable frame and only
  // evaluate it while it is still reasonably fresh (a long no-face gap means the
  // user left the frame, so we stay quiet instead of firing on stale data).
  let latest = null;
  for (let i = observation.frames.length - 1; i >= 0; i -= 1) {
    if (usable(observation.frames[i])) {
      latest = observation.frames[i];
      break;
    }
  }
  if (!latest) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }
  const end = latest.timestampMs;

  // The last valid measurement must not be in the future and must be fresh.
  if (end > context.nowMs || context.nowMs - end > emotionPolicy.maxStaleMs) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }

  const rules = buildRules(z);
  const progress = {};
  let matched = null;
  for (const rule of rules) {
    const start = end - rule.windowMs;
    progress[rule.pattern] = 0;
    if (
      observation.frames[0].timestampMs > start ||
      be >= start ||
      !ready(z, latest, rule.keys)
    ) {
      continue;
    }
    const windowFrames = observation.frames.filter((f) => f.timestampMs >= start);
    const predicate = (f) => usable(f) && ready(z, f, rule.keys) && rule.test(f);
    const persistence = frameRatio(windowFrames, predicate);
    progress[rule.pattern] = persistence;
    if (!matched && persistence >= emotionPolicy.patternPersistence && rule.test(latest)) {
      matched = { rule, persistence };
    }
  }
  if (matched) {
    return {
      action: 'question',
      reason: 'emotion-pattern',
      scenario: matched.rule.scenario,
      pattern: matched.rule.pattern,
      persistence: matched.persistence,
      progress,
    };
  }

  return {
    action: 'quiet',
    reason: 'emotion-only',
    passive: passiveState(z, latest),
    progress,
  };
}

// Record once when the offer was actually shown; dismissal never resets it.
export function recordEmotionOffer(memory, stepId, nowMs) {
  if (memory.offeredStepIds.includes(stepId)) return memory;
  return {
    lastOfferAt: nowMs,
    count: memory.count + 1,
    offeredStepIds: [...memory.offeredStepIds, stepId],
  };
}

// Adapter: map one server session sample to an EmotionFrame. The server emits
// the already-normalized fields (valence [-1,1], arousal [0,1], intensity [0,1],
// category scores [0,1]) plus `valid`. Missing/invalid fields stay NaN so the
// policy rejects the frame instead of fabricating a value.
export function sampleToFrame(sample) {
  const num = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : NaN);
  return {
    timestampMs: num(sample?.at_ms),
    valid: sample?.valid === true,
    valence: num(sample?.valence),
    arousal: num(sample?.arousal),
    intensity: num(sample?.intensity),
    emotions: sample?.emotions && typeof sample.emotions === 'object' ? sample.emotions : {},
  };
}
