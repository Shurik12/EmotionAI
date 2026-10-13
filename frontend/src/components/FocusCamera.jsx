import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { apiClient } from '../api/client';
import { Numa } from './Numa';
import { breakdownTask } from '../utils/taskBreakdown';
import { playFocusCue, unlockFocusSound } from '../utils/focusSound';
import {
  ambientNumaMood,
  baselineReady,
  decideEmotionOffer,
  emotionPatterns,
  emotionPolicy,
  emptyOfferMemory,
  recordEmotionOffer,
  sampleToFrame,
} from '../utils/emotionPolicy';

const FRAME_QUALITY = 0.85;
const MAX_STRIP_SAMPLES = 40;
// The policy needs at least 8 answers in the last 15 s window, so the rate is
// kept at 1–2 s; a slower rate can never satisfy the window.
const SAMPLE_INTERVAL_OPTIONS = [1, 2];
const DEFAULT_SAMPLE_INTERVAL_SEC = 1;
const SESSION_POLL_MS = 1000;
// Keep about a minute of recent frames so the 15 s window is fully covered.
const RECENT_FRAMES = 60;
// Gestures for the answer-flow cards (spec 9–14); help/steps deliberately have
// no movement ("Нума неподвижна").
const CARD_MOODS = Object.freeze({
  'current-action': 'point',
  help: 'rest',
  steps: 'rest',
  pause: 'pause',
  done: 'done',
});
// Minimum time between two ambient Numa gestures, so a noisy emotion signal
// cannot make the assistant flap. Sustained moods still come through.
const AMBIENT_MIN_INTERVAL_MS = 4000;

// Walk backwards from the newest frame while consecutive samples are close
// enough for the policy's `ordered` (max gap 5 s). A camera pause starts a new
// run, so a long gap cannot poison an otherwise usable baseline.
const trailingRun = (frames) => {
  if (frames.length === 0) return [];
  let start = frames.length - 1;
  while (
    start > 0 &&
    frames[start].timestampMs - frames[start - 1].timestampMs <= emotionPolicy.maxGapMs
  ) {
    start -= 1;
  }
  return frames.slice(start);
};

// Camera + emotional dynamics for the Focus session (EMO-17, phase 4).
//
// The camera only starts after an explicit consent checkbox and never records.
// It is owned by this component rather than by the active Focus tab, so moving
// to another Focus tab or browser tab keeps the stream (and the collected
// baseline) alive. While it is on, frames are streamed into a server-side
// session (POST /api/focus/session/<id>/frame) and the normalized observations
// are polled back (GET /api/focus/session/<id>). The server only adapts the core
// output; the `emotion-pilot-v1` decision runs here against a user-confirmed
// personal baseline. Offers stay off until the user enables them, and nothing
// is stored.
export const FocusCamera = ({
  numaNearby = true,
  stepId = null,
  hasStep = false,
  running = false,
  onPoint,
  onSmaller,
  onPause,
  onResume,
  soundEnabled = false,
  soundVolume = 0.7,
  onSoundEnabledChange,
  onSoundVolumeChange,
  currentAction = '',
  onApplySteps,
  doneEvent = null,
}) => {
  const { t } = useLanguage();

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const requestRef = useRef(0);
  const aliveRef = useRef(true);
  const sessionIdRef = useRef(null);
  const updatedRef = useRef(0);
  const inFlightRef = useRef(false);
  const frameUrlRef = useRef(null);
  const offerMemoryRef = useRef(emptyOfferMemory());
  const dismissTimerRef = useRef(null);
  const soundRef = useRef({ enabled: soundEnabled, volume: soundVolume });
  const backCardRef = useRef(null);

  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState('off'); // off | asking | on | error
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [intervalSec, setIntervalSec] = useState(DEFAULT_SAMPLE_INTERVAL_SEC);
  const [message, setMessage] = useState('');
  const [frame, setFrame] = useState(null); // manual snapshot: { url, width, height }
  const [measuring, setMeasuring] = useState(false);
  const [samples, setSamples] = useState([]);
  const [total, setTotal] = useState(0);
  const [offersEnabled, setOffersEnabled] = useState(false);
  const [baseline, setBaseline] = useState(null); // { confirmedOnTask, frames }
  const [card, setCard] = useState(null); // assistant card state machine (spec 1–15)
  const [cardSeq, setCardSeq] = useState(0);
  const [ambientMood, setAmbientMood] = useState('rest');
  const [ambientReaction, setAmbientReaction] = useState(0);
  const [decision, setDecision] = useState(null);
  const ambientMoodRef = useRef('rest');
  const ambientAtRef = useRef(0);

  const intervalMs = intervalSec * 1000;
  const frames = useMemo(() => samples.map(sampleToFrame), [samples]);
  const baselineRun = useMemo(() => trailingRun(frames), [frames]);
  const baselineReadyNow = useMemo(() => baselineReady(baselineRun), [baselineRun]);
  const baselineValid = baselineRun.filter((f) => f.valid).length;
  const baselineSeconds = baselineRun.length
    ? Math.round(
        (baselineRun[baselineRun.length - 1].timestampMs - baselineRun[0].timestampMs) / 1000,
      )
    : 0;

  // Assistant card state machine (spec scenarios 1–15). `lifetimeMs` = 0 means
  // the card waits for the user; the automatic question closes after 20 s.
  const showCard = useCallback((next, lifetimeMs = 0) => {
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    dismissTimerRef.current = null;
    setCard(next);
    setCardSeq((n) => n + 1);
    if (lifetimeMs > 0) {
      dismissTimerRef.current = window.setTimeout(() => {
        dismissTimerRef.current = null;
        setCard(null);
      }, lifetimeMs);
    }
  }, []);

  const dismissCard = useCallback(() => {
    if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    dismissTimerRef.current = null;
    setCard(null);
  }, []);

  const endSession = useCallback(() => {
    const id = sessionIdRef.current;
    sessionIdRef.current = null;
    updatedRef.current = 0;
    inFlightRef.current = false;
    if (id) {
      apiClient.closeFocusSession(id).catch(() => {});
    }
  }, []);

  const startSession = useCallback(async () => {
    try {
      const data = await apiClient.createFocusSession();
      if (!aliveRef.current) return;
      sessionIdRef.current = data.session_id;
      updatedRef.current = 0;
      offerMemoryRef.current = emptyOfferMemory();
      setSamples([]);
      setTotal(0);
      setCard(null);
      setDecision(null);
      setBaseline(null);
    } catch (err) {
      if (aliveRef.current) setMessage(t('focus.camera.sessionError'));
    }
  }, [t]);

  const stopCamera = useCallback(() => {
    requestRef.current += 1;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    dismissCard();
    offerMemoryRef.current = emptyOfferMemory();
    setSamples([]);
    setTotal(0);
    setBaseline(null);
    setDecision(null);
    setStatus('off');
    endSession();
  }, [dismissCard, endSession]);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      stopCamera();
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    };
  }, [stopCamera]);

  const handleStreamEnded = useCallback(() => {
    streamRef.current = null;
    setStatus('error');
    setMessage(t('focus.camera.streamEnded'));
    endSession();
  }, [t, endSession]);

  const enableCamera = async () => {
    if (!consent) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('error');
      setMessage(t('focus.camera.noCamera'));
      return;
    }

    const requestId = ++requestRef.current;
    setStatus('asking');
    setMessage('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      });

      if (requestId !== requestRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      const [track] = stream.getVideoTracks();
      track?.addEventListener('ended', handleStreamEnded);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => setMessage(t('focus.camera.play')));
      }
      setStatus('on');
    } catch (err) {
      if (requestId !== requestRef.current) return;
      setStatus('error');
      const name = err?.name;
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setMessage(t('focus.camera.denied'));
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        setMessage(t('focus.camera.notFound'));
      } else {
        setMessage(t('focus.camera.unavailable'));
      }
    }
  };

  // Grab the current video frame as a JPEG blob, or null if not ready.
  const captureBlob = useCallback(
    () =>
      new Promise((resolve) => {
        const video = videoRef.current;
        if (!video || !video.videoWidth || video.readyState < 2) {
          resolve(null);
          return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(
          (blob) => resolve(blob ? { blob, width: canvas.width, height: canvas.height } : null),
          'image/jpeg',
          FRAME_QUALITY,
        );
      }),
    [],
  );

  const pushFrame = useCallback(
    async (blob) => {
      const id = sessionIdRef.current;
      if (!id) {
        setMessage(t('focus.camera.sessionError'));
        return false;
      }
      inFlightRef.current = true;
      setMeasuring(true);
      // Safety net: never let a lost sample stall the stream forever.
      window.setTimeout(() => {
        inFlightRef.current = false;
      }, Math.max(intervalMs * 3, 4000));
      try {
        await apiClient.sendFocusFrame(id, blob);
        return true;
      } catch (err) {
        inFlightRef.current = false;
        if (aliveRef.current) setMessage(t('focus.camera.error'));
        return false;
      } finally {
        if (aliveRef.current) setMeasuring(false);
      }
    },
    [t, intervalMs],
  );

  // Open the session and read the accumulated observations back.
  useEffect(() => {
    if (status !== 'on') return undefined;

    if (!sessionIdRef.current) startSession();

    const pollId = setInterval(async () => {
      const id = sessionIdRef.current;
      if (!id) return;
      try {
        const session = await apiClient.getFocusSession(id);
        if (!aliveRef.current) return;
        setSamples(session.samples || []);
        setTotal(session.total || session.count || 0);
        if (session.updated_ms && session.updated_ms !== updatedRef.current) {
          updatedRef.current = session.updated_ms;
          inFlightRef.current = false;
        }
      } catch (err) {
        // Transient: the session may be closing. Keep the last known state.
      }
    }, SESSION_POLL_MS);

    return () => clearInterval(pollId);
  }, [status, startSession]);

  // Stream a frame into the session while the camera is on.
  useEffect(() => {
    if (status !== 'on' || !autoAnalyze) return undefined;

    const sampleId = setInterval(async () => {
      if (inFlightRef.current || !sessionIdRef.current) return;
      const shot = await captureBlob();
      if (shot) pushFrame(shot.blob);
    }, intervalMs);

    return () => clearInterval(sampleId);
  }, [status, autoAnalyze, captureBlob, pushFrame, intervalMs]);

  // Run the emotion policy whenever the observation or the context changes.
  // While a card is open no new automatic proposal is made (spec), which also
  // keeps explicit user choices in control.
  useEffect(() => {
    if (status !== 'on' || card) return;
    const nowMs = frames.length ? frames[frames.length - 1].timestampMs : Date.now();
    const observation = {
      source: 'razuma-core',
      // The server adapter maps the core output to the agreed scale
      // (valence [-1,1]; arousal, intensity, categories [0,1]).
      normalizedScaleVerified: true,
      baseline: baseline ?? { confirmedOnTask: false, frames: [] },
      frames: frames.slice(-RECENT_FRAMES),
    };
    const context = {
      session: true,
      visible: !document.hidden,
      running,
      hasCurrentStep: hasStep,
      supportOpen: false,
      enabled: offersEnabled,
      nowMs,
      stepId,
      memory: offerMemoryRef.current,
    };
    const next = decideEmotionOffer(observation, context);
    setDecision(next);
    if (next.action === 'question' && next.pattern && stepId) {
      offerMemoryRef.current = recordEmotionOffer(offerMemoryRef.current, stepId, nowMs);
      // Scenario 15: an unanswered automatic question closes itself after 20 s.
      showCard(
        { kind: 'question', pattern: next.pattern, persistence: next.persistence },
        emotionPolicy.offerLifetimeMs,
      );
    }
  }, [status, frames, baseline, offersEnabled, running, hasStep, stepId, card, showCard]);

  // Keep the sound setting in a ref so the effect deps stay stable.
  useEffect(() => {
    soundRef.current = { enabled: soundEnabled, volume: soundVolume };
  }, [soundEnabled, soundVolume]);

  // Audible cue only for scenario 1 ("Ты сейчас в задаче?"); the spec keeps the
  // other patterns silent (scenarios 2–4). Opt-in.
  useEffect(() => {
    if (card?.kind === 'question' && card.pattern === 'check' && soundRef.current.enabled) {
      playFocusCue('check', soundRef.current.volume);
    }
  }, [card]);

  // Scenario 14 — a completed action (raised from the session): nod + smile
  // gesture and the "ту-ду" tone.
  useEffect(() => {
    if (!doneEvent) return;
    showCard({ kind: 'done', nextAction: doneEvent.nextAction, allDone: doneEvent.allDone }, 6000);
    if (soundRef.current.enabled) playFocusCue('done', soundRef.current.volume);
  }, [doneEvent, showCard]);

  // Ambient reaction (product deviation from rules 5–8): while no card is open,
  // Numa's pose follows the dominant emotion, committed at most every 4 s.
  useEffect(() => {
    if (status !== 'on') {
      if (ambientMoodRef.current !== 'rest') {
        ambientMoodRef.current = 'rest';
        setAmbientMood('rest');
        setAmbientReaction((n) => n + 1);
      }
      return;
    }
    // An open card drives the gesture; keep the ambient mood frozen meanwhile.
    if (card) return;
    const next = ambientNumaMood(frames);
    const nowMs = Date.now();
    if (next !== ambientMoodRef.current && nowMs - ambientAtRef.current >= AMBIENT_MIN_INTERVAL_MS) {
      ambientMoodRef.current = next;
      ambientAtRef.current = nowMs;
      setAmbientMood(next);
      setAmbientReaction((n) => n + 1);
    }
  }, [status, card, frames]);

  const confirmBaseline = () => {
    if (!baselineReadyNow) return;
    setBaseline({ confirmedOnTask: true, frames: baselineRun });
  };

  const handleManualCheck = async () => {
    const shot = await captureBlob();
    if (!shot) {
      setMessage(t('focus.camera.captureHint'));
      return;
    }
    if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
    const url = URL.createObjectURL(shot.blob);
    frameUrlRef.current = url;
    setFrame({ url, width: shot.width, height: shot.height });
    setMessage('');
    await pushFrame(shot.blob);
  };

  // The current action's sub-steps, suggested for scenario 12.
  const suggestedSteps = useMemo(
    () => (currentAction ? breakdownTask(currentAction).map((key) => t(key)) : []),
    [currentAction, t],
  );

  // Answer handling (spec scenarios 9–14). An explicit choice has priority; one
  // reaction is shown and no others are queued.
  const answer = (action) => {
    if (action === 'continue') {
      dismissCard();
      onResume?.();
    } else if (action === 'pause') {
      onPause?.();
      showCard({ kind: 'pause' });
    } else if (action === 'help') {
      showCard({ kind: 'help' });
    } else if (action === 'help-return' || action === 'show-action') {
      onPoint?.();
      showCard({ kind: 'current-action' });
    } else if (action === 'split-steps') {
      onSmaller?.();
      backCardRef.current = card;
      showCard({ kind: 'steps' });
    } else if (action === 'apply-steps') {
      onApplySteps?.(suggestedSteps);
      dismissCard();
    } else if (action === 'edit-steps') {
      dismissCard();
      onPoint?.();
    } else if (action === 'back') {
      showCard(backCardRef.current ?? { kind: 'help' });
    } else if (action === 'resume-session') {
      dismissCard();
      onResume?.();
    }
  };

  const toggleOffers = (value) => {
    setOffersEnabled(value);
    if (!value) dismissCard();
  };

  const strip = samples.slice(-MAX_STRIP_SAMPLES);
  // An open card drives the gesture (spec 1–15); otherwise Numa mirrors the
  // dominant emotion (ambient reaction).
  const numaMood = card
    ? card.kind === 'question'
      ? emotionPatterns[card.pattern]?.mood ?? 'rest'
      : CARD_MOODS[card.kind] ?? 'rest'
    : ambientMood;
  const numaReaction = card ? cardSeq : ambientReaction;

  let statusText = t('focus.emotion.idle');
  if (!total) statusText = t('focus.emotion.noSignal');
  else if (decision?.action === 'quiet' && decision.reason === 'emotion-only') {
    statusText = t('focus.emotion.quiet');
  } else if (decision?.action === 'quiet' && decision.reason === 'insufficient-signal') {
    statusText = t('focus.emotion.unavailable');
  }

  const cardButtons = (actions, keyPrefix) => (
    <div className="focus-offer-actions">
      {actions.map((action, i) => (
        <button
          key={action}
          type="button"
          className={`focus-btn ${i === 0 ? 'primary' : 'ghost'}`}
          onClick={() => answer(action)}
        >
          {t(`${keyPrefix}.${i}`)}
        </button>
      ))}
    </div>
  );

  // The assistant card (spec scenarios 1 and 9–15). Empty card → idle status.
  const renderCard = () => {
    if (!card) return <p className="focus-numa-idle">{statusText}</p>;
    if (card.kind === 'question') {
      const pattern = card.pattern;
      return (
        <>
          <h4>{t(`focus.emotion.patterns.${pattern}.title`)}</h4>
          <p>{t(`focus.emotion.patterns.${pattern}.body`)}</p>
          {cardButtons(
            emotionPatterns[pattern]?.actions ?? ['continue'],
            `focus.emotion.patterns.${pattern}.buttons`,
          )}
        </>
      );
    }
    if (card.kind === 'current-action') {
      return (
        <>
          <h4>{t('focus.emotion.cards.currentActionTitle')}</h4>
          <p>
            {currentAction
              ? t('focus.emotion.cards.currentAction', { action: currentAction })
              : t('focus.emotion.cards.noAction')}
          </p>
          {cardButtons(
            ['continue', 'split-steps', 'pause'],
            'focus.emotion.cards.currentActionButtons',
          )}
        </>
      );
    }
    if (card.kind === 'help') {
      return (
        <>
          <h4>{t('focus.emotion.cards.helpTitle')}</h4>
          {cardButtons(['show-action', 'split-steps', 'pause'], 'focus.emotion.cards.helpButtons')}
        </>
      );
    }
    if (card.kind === 'steps') {
      return (
        <>
          <h4>{t('focus.emotion.cards.stepsTitle')}</h4>
          {suggestedSteps.length > 0 && (
            <ol className="focus-steps">
              {suggestedSteps.map((text, i) => (
                <li key={`${i}-${text}`} className="focus-step">
                  <span className="focus-step-title">{text}</span>
                </li>
              ))}
            </ol>
          )}
          {cardButtons(['apply-steps', 'edit-steps', 'back'], 'focus.emotion.cards.stepsButtons')}
        </>
      );
    }
    if (card.kind === 'pause') {
      return (
        <>
          <h4>{t('focus.emotion.cards.pauseTitle')}</h4>
          {cardButtons(['resume-session'], 'focus.emotion.cards.pauseButtons')}
        </>
      );
    }
    if (card.kind === 'done') {
      return (
        <>
          <h4>{t('focus.emotion.cards.doneTitle')}</h4>
          <p>
            {card.allDone
              ? t('focus.emotion.cards.taskDone')
              : t('focus.emotion.cards.doneNext', { action: card.nextAction || '' })}
          </p>
        </>
      );
    }
    return null;
  };

  return (
    <div className="focus-camera">
      <div className="focus-camera-head">
        <h2>{t('focus.camera.title')}</h2>
        <span className={`focus-camera-state ${status === 'on' ? 'live' : 'off'}`}>
          <span className="focus-camera-dot" aria-hidden="true" />
          {status === 'on' ? t('focus.camera.live') : t('focus.camera.off')}
        </span>
      </div>

      <p className="focus-hint">{t('focus.camera.text')}</p>

      <div className="focus-camera-stage">
        <video
          ref={videoRef}
          className={`focus-camera-video ${status === 'on' ? '' : 'is-hidden'}`}
          autoPlay
          muted
          playsInline
          aria-label={t('focus.camera.previewAria')}
        />
        {status !== 'on' && (
          <div className="focus-camera-placeholder">
            <span className="focus-camera-icon" aria-hidden="true">◎</span>
            <strong>{status === 'asking' ? t('focus.camera.asking') : t('focus.camera.off')}</strong>
          </div>
        )}
      </div>

      <p className="focus-camera-frame-status">
        {measuring
          ? t('focus.camera.measuring')
          : total
            ? t('focus.camera.dynamicCount', { count: total })
            : t('focus.camera.frameNone')}
      </p>

      <label className="focus-option">
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => setConsent(e.target.checked)}
          disabled={status === 'on' || status === 'asking'}
        />
        {t('focus.camera.consent')}
      </label>

      <label className="focus-option">
        <input
          type="checkbox"
          checked={autoAnalyze}
          onChange={(e) => setAutoAnalyze(e.target.checked)}
          disabled={status !== 'on'}
        />
        {t('focus.camera.auto')}
      </label>
      {autoAnalyze && <p className="focus-hint">{t('focus.camera.autoHint')}</p>}
      {autoAnalyze && (
        <label className="focus-option focus-option-select">
          <span>{t('focus.camera.frequency')}</span>
          <select value={intervalSec} onChange={(e) => setIntervalSec(Number(e.target.value))}>
            {SAMPLE_INTERVAL_OPTIONS.map((sec) => (
              <option key={sec} value={sec}>
                {t('focus.camera.seconds', { n: sec })}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="focus-option">
        <input
          type="checkbox"
          checked={offersEnabled}
          onChange={(e) => toggleOffers(e.target.checked)}
        />
        {t('focus.emotion.offersLabel')}
      </label>
      <p className="focus-hint">{t('focus.emotion.offersHint')}</p>

      <label className="focus-option">
        <input
          type="checkbox"
          checked={soundEnabled}
          onChange={(e) => {
            onSoundEnabledChange?.(e.target.checked);
            if (e.target.checked) unlockFocusSound();
          }}
        />
        {t('focus.sound.enable')}
      </label>
      {soundEnabled && (
        <div className="focus-sound">
          <label className="focus-option focus-option-select">
            <span>{t('focus.sound.volume')}</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={soundVolume}
              onChange={(e) => onSoundVolumeChange?.(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="focus-btn ghost"
            onClick={() => {
              unlockFocusSound();
              playFocusCue('check', soundVolume);
            }}
          >
            {t('focus.sound.preview')}
          </button>
        </div>
      )}
      <p className="focus-hint">{t('focus.sound.hint')}</p>

      {status === 'on' && (
        <div className="focus-baseline">
          <div className="focus-baseline-head">
            <span className="focus-baseline-label">{t('focus.emotion.baselineTitle')}</span>
            <span
              className={`focus-baseline-badge ${
                baseline?.confirmedOnTask ? 'confirmed' : baselineReadyNow ? 'ready' : ''
              }`}
            >
              {baseline?.confirmedOnTask
                ? t('focus.emotion.baselineConfirmed')
                : `${baselineValid} · ${baselineSeconds}s`}
            </span>
          </div>
          {!baseline?.confirmedOnTask && (
            <>
              <p className="focus-hint">
                {baselineReadyNow
                  ? t('focus.emotion.baselineReady')
                  : baselineValid === 0
                    ? t('focus.emotion.noSignal')
                    : t('focus.emotion.baselineCollecting', {
                        valid: baselineValid,
                        seconds: baselineSeconds,
                      })}
              </p>
              <button
                type="button"
                className="focus-btn ghost"
                onClick={confirmBaseline}
                disabled={!baselineReadyNow}
                title={
                  baselineReadyNow
                    ? undefined
                    : t('focus.emotion.baselineCollecting', {
                        valid: baselineValid,
                        seconds: baselineSeconds,
                      })
                }
              >
                {t('focus.emotion.baselineConfirm')}
              </button>
            </>
          )}
        </div>
      )}

      <div className="focus-actions">
        {status === 'on' ? (
          <>
            <button type="button" className="focus-btn primary" onClick={handleManualCheck} disabled={measuring}>
              {measuring ? t('focus.camera.measuring') : t('focus.camera.capture')}
            </button>
            <button type="button" className="focus-btn ghost" onClick={stopCamera}>
              {t('focus.camera.disable')}
            </button>
          </>
        ) : (
          <button
            type="button"
            className="focus-btn primary"
            onClick={enableCamera}
            disabled={!consent || status === 'asking'}
          >
            {status === 'asking' ? t('focus.camera.asking') : t('focus.camera.enable')}
          </button>
        )}
      </div>

      {message && (
        <p className="focus-camera-message" role="status">
          {message}
        </p>
      )}

      {frame && (
        <>
          <p className="focus-camera-frame-status">
            {t('focus.camera.frameReady', { width: frame.width, height: frame.height })}
          </p>
          <figure className="focus-camera-snapshot">
            <img src={frame.url} alt={t('focus.camera.frameAlt')} />
          </figure>
        </>
      )}

      <div
        className={`focus-numa ${card?.kind === 'question' ? 'offer' : 'idle'}`}
        aria-live="polite"
      >
        {numaNearby && (
          <Numa small mood={numaMood} reaction={numaReaction} />
        )}
        <p className="focus-numa-label">{numaNearby ? t('focus.emotion.numa') : t('focus.emotion.core')}</p>
        {renderCard()}
      </div>

      <div className="focus-dynamics">
        <p className="focus-dynamics-title">
          {total
            ? t('focus.camera.dynamicCount', { count: total })
            : t('focus.camera.dynamic')}
        </p>
        {strip.length ? (
          <div className="focus-dynamics-bars" aria-hidden="true">
            {strip.map((sample, index) => (
              <span
                key={`${index}-${sample.at_ms}`}
                className={`focus-dynamics-bar ${sample.valid ? 'valid' : 'invalid'}`}
                style={{ height: `${15 + (sample.arousal ?? 0.5) * 85}%` }}
              />
            ))}
          </div>
        ) : (
          <p className="focus-hint">{t('focus.camera.dynamicEmpty')}</p>
        )}
      </div>

      <p className="focus-camera-note">{t('focus.emotion.note')}</p>
    </div>
  );
};
