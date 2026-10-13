// Focus assistant cues, synthesized on demand via the Web Audio API. The
// timbres follow the v6 spec so no audio files are shipped:
//   - "check" (a question is shown): one 440 Hz tone, 400 ms
//     (attack 80 ms, hold 120 ms, decay 200 ms).
//   - "done" (an action is completed): "ту-ду" — 523 Hz 160 ms, 60 ms gap,
//     659 Hz 220 ms (attack 30 ms, decay 80 ms each).
// The startup digital peak is −24 dBFS; the user volume scales it and nothing
// is auto-amplified. Playback is opt-in and does nothing until a user gesture
// has unlocked the audio context.

const PEAK_DBFS = -24;
const PEAK = 10 ** (PEAK_DBFS / 20);

const CHECK = [{ frequency: 440, duration: 0.4, attack: 0.08, decay: 0.2, offset: 0 }];
const DONE = [
  { frequency: 523, duration: 0.16, attack: 0.03, decay: 0.08, offset: 0 },
  { frequency: 659, duration: 0.22, attack: 0.03, decay: 0.08, offset: 0.22 },
];

let context = null;

const audioContext = () => {
  if (typeof window === 'undefined') return null;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  if (!context) context = new Ctor();
  return context;
};

// Create/resume the context from a user gesture (toggling sound, previewing).
export const unlockFocusSound = () => {
  const ctx = audioContext();
  if (ctx && ctx.state === 'suspended') ctx.resume().catch(() => {});
};

const startTone = (ctx, { frequency, duration, attack, decay, at, gain }) => {
  const osc = ctx.createOscillator();
  const amp = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(frequency, at);
  amp.gain.setValueAtTime(0, at);
  amp.gain.linearRampToValueAtTime(gain, at + attack);
  amp.gain.setValueAtTime(gain, at + duration - decay);
  amp.gain.linearRampToValueAtTime(0, at + duration);
  osc.connect(amp);
  amp.connect(ctx.destination);
  osc.start(at);
  osc.stop(at + duration);
};

/**
 * Play a cue. `kind` is 'check' (question) or 'done' (completion); `volume`
 * is the user setting in [0, 1].
 */
export const playFocusCue = (kind, volume = 1) => {
  const ctx = audioContext();
  if (!ctx) return false;
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  const gain = PEAK * Math.min(Math.max(volume, 0), 1);
  if (gain <= 0) return false;
  const start = ctx.currentTime + 0.02;
  for (const tone of kind === 'done' ? DONE : CHECK) {
    startTone(ctx, { ...tone, at: start + tone.offset, gain });
  }
  return true;
};
