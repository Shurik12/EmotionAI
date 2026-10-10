import React from 'react';

// Numa — the Razuma Focus character (EMO-17).
//
// Rendered from the flat cutout layers exported by the animation prototypes
// (`focus-assets/assets/*.png`, copied to `public/static/numa/`). Each layer is
// a separate <img> positioned in the 760×480 prototype canvas; the gesture
// classes drive the transforms defined in `styles/components/focus.css`.
//
// `gestureKey` is bumped by the caller on every new reaction so the CSS
// animation restarts (React remounts the subtree). When `animate` is false the
// character stays completely still, as required in the no-animation mode.
// `returning` switches the layers to the short "release" animation when the
// card closes, so a held pose eases back instead of snapping.
export const NumaCharacter = ({
  gesture = null,
  gestureKey = 0,
  animate = true,
  returning = false,
  className = '',
}) => {
  const active = animate ? gesture || 'idle' : 'idle';
  return (
    <div
      key={gestureKey}
      className={`numa-character ${className}`}
      data-gesture={active}
      data-phase={returning ? 'return' : 'enter'}
      aria-hidden="true"
    >
      <img className="numa-layer numa-body" src="/static/numa/body.png" alt="" draggable="false" />
      <img className="numa-layer numa-left" src="/static/numa/left.png" alt="" draggable="false" />
      <img className="numa-layer numa-right" src="/static/numa/right.png" alt="" draggable="false" />
      <img className="numa-layer numa-head" src="/static/numa/head.png" alt="" draggable="false" />
      <img className="numa-layer numa-happy" src="/static/numa/happy.png" alt="" draggable="false" />
    </div>
  );
};
