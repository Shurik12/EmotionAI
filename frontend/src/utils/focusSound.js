// Razuma Focus audio cues (EMO-17).
//
// Only two scenarios have sound (rows 1 and 14 of
// `docs/focus/FOCUS_INTERVENTION_SPEC.md`), both synthesized with the WebAudio
// API so no binary asset is required:
//   - `tu`    — one 440 Hz sine, 400 ms (attack 80 / hold 120 / release 200).
//   - `tu-du` — 523 Hz 160 ms, 60 ms pause, 659 Hz 220 ms (attack 30 / release 80).
//
// Both start at a fixed peak of −24 dBFS and are never amplified
// automatically; the user controls a volume and can preview each cue.

const PEAK_DBFS = -24;
const PEAK = 10 ** (PEAK_DBFS / 20); // ≈ 0.0631

let ctx = null;

const getContext = () => {
  if (typeof window === 'undefined') return null;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  if (!ctx) ctx = new Ctor();
  return ctx;
};

// One sine note with an explicit linear envelope. `attack + release` must fit
// inside `duration`; the remainder is the sustain.
function tone(context, { freq, startAt, duration, attack, release, volume }) {
  const safeAttack = Math.min(attack, duration);
  const safeRelease = Math.min(release, Math.max(0, duration - safeAttack));
  const peak = PEAK * volume;
  const osc = context.createOscillator();
  const gain = context.createGain();
  osc.type = 'sine';
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0, startAt);
  gain.gain.linearRampToValueAtTime(peak, startAt + safeAttack);
  gain.gain.setValueAtTime(peak, startAt + duration - safeRelease);
  gain.gain.linearRampToValueAtTime(0, startAt + duration);
  osc.connect(gain);
  gain.connect(context.destination);
  osc.start(startAt);
  osc.stop(startAt + duration + 0.02);
}

function schedule(context, kind, volume) {
  const t0 = context.currentTime + 0.02;
  if (kind === 'tu') {
    tone(context, { freq: 440, startAt: t0, duration: 0.4, attack: 0.08, release: 0.2, volume });
  } else if (kind === 'tu-du') {
    tone(context, { freq: 523, startAt: t0, duration: 0.16, attack: 0.03, release: 0.08, volume });
    tone(context, {
      freq: 659,
      startAt: t0 + 0.16 + 0.06,
      duration: 0.22,
      attack: 0.03,
      release: 0.08,
      volume,
    });
  }
}

// Play a scenario cue. `enabled` is the user preference for that cue.
export function playFocusSound(kind, { enabled = true, volume = 0.7 } = {}) {
  if (!enabled || !kind || volume <= 0) return;
  const context = getContext();
  if (!context) return;
  const run = () => schedule(context, kind, volume);
  if (context.state === 'suspended') context.resume().then(run).catch(() => {});
  else run();
}

// Preview ignores the on/off preference (it is an explicit user action) but
// still respects the volume.
export function previewFocusSound(kind, volume = 0.7) {
  const context = getContext();
  if (!context) return;
  const run = () => schedule(context, kind, volume);
  if (context.state === 'suspended') context.resume().then(run).catch(() => {});
  else run();
}
