// Local, deterministic mapping from the Razuma core response (face analysis)
// to a calm "next step" hint for the Focus page (EMO-17).
//
// No LLM and no extra calls: the existing face endpoint already returns
// per-emotion probabilities plus valence/arousal (see src/emotionai/Image.cpp).
// This module turns that signal into translation KEYS (not human-readable
// strings), the same "content as keys" approach as taskBreakdown.js and
// BurnoutAnalyzer — the UI resolves them through t().

const NEGATIVE_EMOTIONS = ['anger', 'fear', 'sad', 'disgust', 'contempt'];

export const clamp01 = (value) => Math.min(1, Math.max(0, value));

const toNumber = (value) => {
  const num = parseFloat(value);
  return Number.isFinite(num) ? num : null;
};

// additional_probs arrives as decimal strings: emotions + valence/arousal.
// A missing or empty payload means the core found no usable face signal.
export const summarizeSignal = (result) => {
  const probs = result?.additional_probs;
  if (!probs || typeof probs !== 'object' || !Object.keys(probs).length) return null;

  const emotions = {};
  Object.entries(probs).forEach(([key, value]) => {
    if (key === 'valence' || key === 'arousal') return;
    const num = toNumber(value);
    if (num !== null) emotions[key] = num;
  });

  if (!Object.keys(emotions).length) return null;

  const neutral = emotions.neutral ?? 0;
  const negative = NEGATIVE_EMOTIONS.reduce((sum, key) => sum + (emotions[key] || 0), 0);
  const explicitArousal = toNumber(probs.arousal);

  return {
    emotions,
    // enet_b2_7 has no arousal output: fall back to "1 - neutral".
    arousal: clamp01(explicitArousal ?? Math.max(0, 1 - neutral)),
    negative,
    hasFace: true,
  };
};

const REACTION_KEYS = {
  noSignal: 'focus.reaction.noSignal',
  steady: 'focus.reaction.steady',
  rising: 'focus.reaction.rising',
};

// High activation or a heavy negative load is treated as "maybe the step is
// too big" — an invitation to clarify, never a verdict about focus.
export const buildCameraReaction = (result) => {
  const signal = summarizeSignal(result);

  let kind = 'steady';
  if (!signal || !signal.hasFace) {
    kind = 'noSignal';
  } else if (signal.arousal >= 0.6 || signal.negative >= 0.5) {
    kind = 'rising';
  }

  return { kind, baseKey: REACTION_KEYS[kind], signal };
};
