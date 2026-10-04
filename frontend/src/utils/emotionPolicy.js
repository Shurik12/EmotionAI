// Faithful JavaScript port of the Razuma Focus v6 emotion policy
// (`source/lib/emotion-policy.ts`, profile `emotion-pilot-v1`) from
// Razuma_Focus_Developer_v6_Emotion_Rules_2026-10-04.
//
// The supplied policy is a pure function over a normalized observation:
//   - `EmotionFrame` = { timestampMs, valid, valence, arousal, intensity, emotions }
//     with valence in [-1, 1] and arousal / intensity / category scores in [0, 1].
//   - `EmotionObservation` = { source, normalizedScaleVerified, baseline:{confirmedOnTask,frames}, frames }.
// The EmotionAI adapter (src/focus/FocusSessionManager) produces the normalized
// frames from the core response; this module owns the decision logic.
//
// Every threshold is a pilot hypothesis, not a validated attention classifier.
// The review only motivates caution, a personal baseline and temporal
// persistence; it does not validate these combinations or numbers.

export const emotionPolicy = Object.freeze({
  version: 'emotion-pilot-v1',
  baselineMs: 60000,
  windowMs: 15000,
  persistence: 0.8,
  deviation: 2,
  scaleFloor: 0.05,
  maxGapMs: 5000,
  cooldownMs: 300000,
  maxOffersPerSession: 2,
  offerLifetimeMs: 20000,
});

export const emptyOfferMemory = () => ({ lastOfferAt: null, offeredStepIds: [], count: 0 });

// UI metadata for the three pilot patterns. The human-readable copy lives in
// the translations under `focus.emotion.patterns.<pattern>`, keyed by the same
// id, so the same algorithm serves RU and EN.
export const emotionPatterns = Object.freeze({
  'fear-high': { mood: 'support', primary: 'pause' },
  'friction-high': { mood: 'support', primary: 'smaller' },
  'sadness-low': { mood: 'point', primary: 'point' },
});

const median = (values) => {
  const a = [...values].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
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

// Duration-weighted ratio, so a dense burst cannot outweigh a longer window.
// Only intervals with usable endpoints satisfying the predicate count.
function durationRatio(frames, start, end, predicate) {
  let ms = 0;
  for (let i = 1; i < frames.length; i++) {
    const a = frames[i - 1];
    const b = frames[i];
    const dt = Math.max(0, Math.min(end, b.timestampMs) - Math.max(start, a.timestampMs));
    if (usable(a) && usable(b) && predicate(a) && predicate(b)) ms += dt;
  }
  return ms / (end - start);
}

// Mirrors the baseline checks in decideEmotionOffer (lines 42-44 of the spec)
// so the UI can enable "confirm baseline" only when it will actually be accepted.
export function baselineReady(frames) {
  if (!Array.isArray(frames) || frames.length < 16 || !ordered(frames)) return false;
  const start = frames[0].timestampMs;
  const end = frames[frames.length - 1].timestampMs;
  return (
    end - start >= emotionPolicy.baselineMs &&
    durationRatio(frames, start, end, () => true) >= emotionPolicy.persistence
  );
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
    baseline.frames.length < 16 ||
    !ordered(baseline.frames)
  ) {
    return { action: 'quiet', reason: 'baseline-required' };
  }
  const bs = baseline.frames[0].timestampMs;
  const be = baseline.frames[baseline.frames.length - 1].timestampMs;
  if (
    be - bs < emotionPolicy.baselineMs ||
    durationRatio(baseline.frames, bs, be, () => true) < emotionPolicy.persistence
  ) {
    return { action: 'quiet', reason: 'baseline-required' };
  }
  if (!Array.isArray(observation.frames) || observation.frames.length < 8 || !ordered(observation.frames)) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }
  const end = observation.frames[observation.frames.length - 1].timestampMs;
  const start = end - emotionPolicy.windowMs;
  if (
    end > context.nowMs ||
    context.nowMs - end > emotionPolicy.maxGapMs ||
    observation.frames[0].timestampMs > start ||
    be >= start
  ) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }
  if (durationRatio(observation.frames, start, end, () => true) < emotionPolicy.persistence) {
    return { action: 'quiet', reason: 'insufficient-signal' };
  }
  const validBaseline = baseline.frames.filter(usable);
  const read = (f, k) => (k === 'valence' || k === 'arousal' || k === 'intensity' ? f[k] : f.emotions[k]);
  const scales = new Map();
  const z = (f, k) => {
    const value = read(f, k);
    if (value === undefined) return NaN;
    let s = scales.get(k);
    if (!s) {
      const a = validBaseline.map((x) => read(x, k)).filter((x) => x !== undefined);
      if (a.length < 16 || a.length / validBaseline.length < emotionPolicy.persistence) return NaN;
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
    return (value - s.center) / s.scale;
  };
  const d = emotionPolicy.deviation;
  const high = (f) => f.valence < 0 && z(f, 'valence') <= -d && z(f, 'arousal') >= d && z(f, 'intensity') >= d;
  const rules = [
    [
      'fear-high',
      (f) =>
        high(f) &&
        z(f, 'fear') >= d &&
        z(f, 'fear') >=
          Math.max(
            Number.isFinite(z(f, 'anger')) ? z(f, 'anger') : -Infinity,
            Number.isFinite(z(f, 'disgust')) ? z(f, 'disgust') : -Infinity,
          ),
    ],
    ['friction-high', (f) => high(f) && (z(f, 'anger') >= d || z(f, 'disgust') >= d)],
    ['sadness-low', (f) => f.valence < 0 && z(f, 'valence') <= -d && z(f, 'arousal') <= -d && z(f, 'sadness') >= d],
  ];
  const latest = observation.frames[observation.frames.length - 1];
  for (const [pattern, test] of rules) {
    const persistence = durationRatio(observation.frames, start, end, test);
    if (usable(latest) && test(latest) && persistence >= emotionPolicy.persistence) {
      return { action: 'question', reason: 'emotion-pattern', pattern, persistence };
    }
  }
  return { action: 'quiet', reason: 'emotion-only' };
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
