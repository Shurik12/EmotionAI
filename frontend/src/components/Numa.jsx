import React, { useEffect, useMemo, useRef } from 'react';
import { useLanguage } from '../hooks/useLanguage';

// Numa (ОРИ) — the Razuma assistant. A 2D cutout rig assembled from the layer
// artwork in focus-assets (assets/head|body|left|right|happy.png, see
// /home/alex/Downloads/for_emotion_ai). The geometry below mirrors the
// prototype compositor (build_animations.py) which lays the layers out on a
// 760 × 480 canvas and rotates them about shoulder/neck pivots. Each gesture
// plays once per event and settles back to rest; quiet states stay static.
// The demo canvas is mostly empty; crop to the character. The box is wide
// enough for the fully extended arms of the "support"/"point" gestures.
const CROP = { x: 14, y: 100, w: 352, h: 328 };

const SHADOW = { x: 115, y: 406, w: 144, h: 15 };

// Static layer boxes on the reference canvas: [x, y] of the pivot plus the
// drawn size and the pivot offset inside the layer (matches build_animations.py).
const BODY = { pos: [190, 402], size: [124, 178], anchor: [62, 178] };
const ARM_LEFT = { pos: [134, 262], size: [82, 108], anchor: [67, 15] };
const ARM_RIGHT = { pos: [246, 262], size: [82, 108], anchor: [15, 15] };
const HEAD = { pos: [190, 273], size: [220, 159], anchor: [110, 159] };

const restPose = () => ({ head: 0, lift: 0, left: 0, right: 0, nod: 0, happy: 0 });

const ease = (x) => {
  const v = Math.max(0, Math.min(1, x));
  return v * v * (3 - 2 * v);
};

// Ramp in, hold, ramp out — the prototype's `pulse`.
const pulse = (t, a, b, c) => (t < a ? ease(t / a) : t < b ? 1 : 1 - ease((t - b) / (c - b)));

// Gesture poses, ported 1:1 from build_animations.py (angles in degrees).
const poseAt = (gesture, t) => {
  const p = restPose();
  if (gesture === 'check') {
    const k = pulse(t, 450, 650, 900);
    p.head = 6 * k;
    p.lift = -6 * k;
    p.left = -102 * k;
  } else if (gesture === 'help') {
    p.head = 5 * pulse(t, 250, 450, 700);
  } else if (gesture === 'support') {
    const k = pulse(t, 300, 500, 900);
    p.left = 62 * k;
    p.right = -62 * k;
  } else if (gesture === 'point') {
    const k = pulse(t, 400, 550, 800);
    p.head = 5 * k;
    p.right = -72 * k;
  } else if (gesture === 'pause') {
    const k = ease(t / 600);
    p.nod = 5 * k;
    p.left = -8 * k;
    p.right = 8 * k;
  } else if (gesture === 'done') {
    p.nod = 7 * pulse(t, 200, 200, 450);
    p.happy = ease((t - 450) / 200);
  }
  return p;
};

const DURATIONS = Object.freeze({
  still: 0,
  check: 900,
  help: 700,
  support: 900,
  point: 800,
  pause: 600,
  done: 650,
});

// Focus-page moods map onto the prototype's scenario gestures.
export const gestures = Object.freeze({
  idle: 'still',
  rest: 'still',
  focus: 'still',
  thinking: 'help',
  help: 'help',
  check: 'check',
  support: 'support',
  point: 'point',
  happy: 'done',
  done: 'done',
  pause: 'pause',
});

const pct = (value, total) => `${(value / total) * 100}%`;

// A layer is positioned so that `pos` lands under its `anchor`, then rotated
// about that anchor. CSS rotate(+deg) is clockwise, matching PIL rotate(-deg).
const layerStyle = (pos, size, anchor, angle = 0) => ({
  left: pct(pos[0] - anchor[0] - CROP.x, CROP.w),
  top: pct(pos[1] - anchor[1] - CROP.y, CROP.h),
  width: pct(size[0], CROP.w),
  height: pct(size[1], CROP.h),
  transformOrigin: `${pct(anchor[0], size[0])} ${pct(anchor[1], size[1])}`,
  transform: `rotate(${angle}deg)`,
});

const headStyle = (p) => {
  const h = HEAD.size[1] * (1 - p.nod * 0.009);
  const lift = p.lift + p.nod * 0.7;
  return layerStyle([HEAD.pos[0], HEAD.pos[1] + lift], [HEAD.size[0], h], [HEAD.anchor[0], h], p.head);
};

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export const Numa = ({
  small = false,
  mood = 'idle',
  reaction = 0,
  motion = true,
  onGreet,
  className = '',
}) => {
  const { t } = useLanguage();
  const leftRef = useRef(null);
  const rightRef = useRef(null);
  const headRef = useRef(null);
  const happyRef = useRef(null);
  const rafRef = useRef(0);
  const timerRef = useRef(0);

  const gesture = gestures[mood] || 'still';

  const applyPose = (p) => {
    const head = headStyle(p);
    if (headRef.current) Object.assign(headRef.current.style, head);
    if (happyRef.current) {
      Object.assign(happyRef.current.style, head);
      happyRef.current.style.opacity = String(p.happy);
    }
    if (leftRef.current) {
      Object.assign(leftRef.current.style, layerStyle(ARM_LEFT.pos, ARM_LEFT.size, ARM_LEFT.anchor, p.left));
    }
    if (rightRef.current) {
      Object.assign(rightRef.current.style, layerStyle(ARM_RIGHT.pos, ARM_RIGHT.size, ARM_RIGHT.anchor, p.right));
    }
  };

  const stop = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timerRef.current) window.clearTimeout(timerRef.current);
    rafRef.current = 0;
    timerRef.current = 0;
  };

  const play = (name) => {
    stop();
    const duration = DURATIONS[name] || 0;
    if (!motion || !duration || prefersReducedMotion()) {
      applyPose(restPose());
      return;
    }
    const startedAt = performance.now();
    const tick = (now) => {
      const elapsed = now - startedAt;
      applyPose(poseAt(name, Math.min(elapsed, duration)));
      if (elapsed < duration) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        // Hold the final frame briefly (the "done" smile), then relax.
        timerRef.current = window.setTimeout(() => applyPose(restPose()), 700);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  useEffect(() => {
    play(gesture);
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gesture, reaction, motion]);

  // Initial frame so the rig is correct before the first effect runs.
  const rest = useMemo(() => {
    const p = restPose();
    return {
      shadow: {
        left: pct(SHADOW.x - CROP.x, CROP.w),
        top: pct(SHADOW.y - CROP.y, CROP.h),
        width: pct(SHADOW.w, CROP.w),
        height: pct(SHADOW.h, CROP.h),
      },
      body: layerStyle(BODY.pos, BODY.size, BODY.anchor, 0),
      left: layerStyle(ARM_LEFT.pos, ARM_LEFT.size, ARM_LEFT.anchor, 0),
      right: layerStyle(ARM_RIGHT.pos, ARM_RIGHT.size, ARM_RIGHT.anchor, 0),
      head: headStyle(p),
      happy: { ...headStyle(p), opacity: 0 },
    };
  }, []);

  return (
    <button
      type="button"
      className={`numa ${small ? 'small' : ''} ${className}`.trim()}
      data-mood={mood}
      data-motion={motion}
      aria-label={t('focus.numa.greet')}
      title={t('focus.numa.greetTitle')}
      onClick={() => {
        play('check');
        onGreet?.();
      }}
    >
      <span className="ori-rig" aria-hidden="true">
        <span className="ori-layer ori-shadow" style={rest.shadow} />
        <span className="ori-layer ori-body" style={rest.body} />
        <span className="ori-layer ori-left" ref={leftRef} style={rest.left} />
        <span className="ori-layer ori-right" ref={rightRef} style={rest.right} />
        <span className="ori-layer ori-head" ref={headRef} style={rest.head} />
        <span className="ori-layer ori-happy" ref={happyRef} style={rest.happy} />
      </span>
    </button>
  );
};
