import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { apiClient } from '../api/client';
import {
  baselineCoverage,
  baselineReady,
  decideEmotionOffer,
  emotionPolicy,
  emptyOfferMemory,
  recordEmotionOffer,
  sampleToFrame,
} from '../utils/emotionPolicy';
import {
  DISMISS_MS,
  TIMEOUT_DISMISS_MS,
  TIMEOUT_MS,
  answerRouting,
  resolveIntervention,
} from '../utils/focusScenarios';
import { playFocusSound, previewFocusSound } from '../utils/focusSound';
import { NumaCharacter } from './NumaCharacter';

const FRAME_QUALITY = 0.85;
const MAX_STRIP_SAMPLES = 40;
// The policy needs at least 8 answers in the decision window, so the rate is
// kept at 1–2 s; a slower rate can never satisfy the window.
const SAMPLE_INTERVAL_OPTIONS = [1, 2];
const DEFAULT_SAMPLE_INTERVAL_SEC = 1;
const SESSION_POLL_MS = 1000;
// Keep about a minute of recent frames so the 15 s window is fully covered.
const RECENT_FRAMES = 60;

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

// Camera + emotional dynamics for the Focus session (EMO-17).
//
// The camera only starts after an explicit consent checkbox, stops as soon as
// the tab is hidden, and never records. While it is on, frames are streamed
// into a server-side session (POST /api/focus/session/<id>/frame) and the
// normalized observations are polled back (GET /api/focus/session/<id>). The
// server only adapts the core output; the 16-scenario decision (emotion-pilot-v2)
// runs here against a user-confirmed personal baseline. Offers stay off until
// the user enables them, and nothing is stored.
export const FocusCamera = ({
  numaNearby = true,
  stepId = null,
  hasStep = false,
  running = false,
  stepLabel = '',
  nextStepLabel = '',
  steps = [],
  allDone = false,
  doneSignal = 0,
  onPoint,
  onSmaller,
  onPause,
  onResume,
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
  const interventionTimerRef = useRef(null);
  const lastDoneRef = useRef(0);
  const lastReasonRef = useRef('');

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
  const [intervention, setIntervention] = useState(null); // { id, payload }
  const [gestureKey, setGestureKey] = useState(0);
  const [decision, setDecision] = useState(null);
  const [answerLog, setAnswerLog] = useState([]);
  // Sound and motion preferences (spec: adjustable volume, preview, no
  // auto-amplification; a no-animation mode keeps static cards only).
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [soundVolume, setSoundVolume] = useState(0.7);
  const [motionEnabled, setMotionEnabled] = useState(true);

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
  const baselineCoveragePct = Math.round(baselineCoverage(baselineRun) * 100);

  const soundOptions = { enabled: soundEnabled, volume: soundVolume };
  // `?debug` reveals the gating state, the last policy reason and manual
  // gesture previews (useful while validating the animations).
  const debug =
    typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('debug');

  const clearInterventionTimers = useCallback(() => {
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
    if (interventionTimerRef.current) {
      clearTimeout(interventionTimerRef.current);
      interventionTimerRef.current = null;
    }
  }, []);

  // Open a scenario card. `timeout` controls the row-15 auto-dismiss (only
  // automatic offers wait 20 s before recording an unknown answer).
  const openIntervention = useCallback(
    (id, payload = {}, { timeout = false } = {}) => {
      const def = resolveIntervention(id);
      if (!def) return;
      clearInterventionTimers();
      setIntervention({ id, payload });
      setGestureKey((k) => k + 1);
      if (def.sound) playFocusSound(def.sound, soundOptions);
      if (timeout) {
        interventionTimerRef.current = window.setTimeout(() => {
          interventionTimerRef.current = null;
          setAnswerLog((log) => [...log, { id, answer: null, at: Date.now() }]);
          // Row 15: the card disappears over 200 ms, no repeated gesture or sound.
          closeIntervention(TIMEOUT_DISMISS_MS);
        }, TIMEOUT_MS);
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [clearInterventionTimers, soundEnabled, soundVolume],
  );

  const closeIntervention = useCallback(
    (dismissMs = DISMISS_MS) => {
      clearInterventionTimers();
      if (dismissMs > 0) {
        // Let the outgoing-card transition play before unmounting.
        dismissTimerRef.current = window.setTimeout(() => {
          dismissTimerRef.current = null;
          setIntervention(null);
        }, dismissMs);
      } else {
        setIntervention(null);
      }
    },
    [clearInterventionTimers],
  );

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
      setIntervention(null);
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
    clearInterventionTimers();
    offerMemoryRef.current = emptyOfferMemory();
    setSamples([]);
    setTotal(0);
    setBaseline(null);
    setDecision(null);
    setIntervention(null);
    setStatus('off');
    endSession();
  }, [clearInterventionTimers, endSession]);

  useEffect(() => {
    aliveRef.current = true;
    const handleVisibility = () => {
      if (document.hidden) stopCamera();
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      aliveRef.current = false;
      stopCamera();
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
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

      if (requestId !== requestRef.current || document.hidden) {
        stream.getTracks().forEach((track) => track.stop());
        if (requestId === requestRef.current) setStatus('off');
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
  useEffect(() => {
    if (status !== 'on' || intervention) return;
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
    // Lightweight observability: which row fired, or why Numa stayed still.
    if (next.action === 'question') {
      console.debug(
        `[focus] offer #${next.scenario} (${next.pattern}) persistence=${Number(next.persistence).toFixed(2)}`,
      );
    } else if (next.reason !== lastReasonRef.current) {
      lastReasonRef.current = next.reason;
      console.debug(`[focus] quiet: ${next.reason}${next.passive ? ` (${next.passive})` : ''}`);
    }
    if (next.action === 'question' && next.pattern && stepId) {
      offerMemoryRef.current = recordEmotionOffer(offerMemoryRef.current, stepId, nowMs);
      openIntervention(next.pattern, {}, { timeout: true });
    }
  }, [status, frames, baseline, offersEnabled, running, hasStep, stepId, intervention, openIntervention]);

  // Scenario 14: a confirmed completion is a one-time reaction.
  useEffect(() => {
    if (!doneSignal || doneSignal === lastDoneRef.current) return;
    lastDoneRef.current = doneSignal;
    openIntervention('done', { allDone, action: nextStepLabel }, { timeout: false });
  }, [doneSignal, allDone, nextStepLabel, openIntervention]);

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

  // Explicit user choices always win over the automatic rules; the follow-up
  // card comes from `answerRouting` (rows 9–13 of the spec).
  const answer = (value) => {
    setAnswerLog((log) => [...log, { id: intervention?.id ?? null, answer: value, at: Date.now() }]);
    if (value === 'pause') onPause?.();
    else if (value === 'resume') onResume?.();
    else if (value === 'show_current_action' || value === 'help_return') onPoint?.();
    else if (value === 'edit_steps') onSmaller?.();

    const next = answerRouting[value];
    if (value === 'continue' || value === 'use_steps' || value === 'edit_steps') {
      closeIntervention(DISMISS_MS); // row 10
      return;
    }
    if (next) openIntervention(next, { action: stepLabel });
    else closeIntervention(DISMISS_MS);
  };

  // A manually requested help card stays available even without an offer.
  const requestHelp = () => openIntervention('helpMenu', { action: stepLabel });

  const toggleOffers = (value) => {
    setOffersEnabled(value);
    if (!value) closeIntervention(0);
  };

  const def = intervention ? resolveIntervention(intervention.id) : null;
  const gesture = def?.gesture ?? null;
  const strip = samples.slice(-MAX_STRIP_SAMPLES);
  const showUnavailable =
    status === 'on' && !intervention && total > 0 && baselineValid === 0;

  let statusText = t('focus.emotion.idle');
  if (!total) statusText = t('focus.emotion.noSignal');
  else if (decision?.action === 'quiet' && decision.reason === 'emotion-only') {
    statusText = t('focus.emotion.quiet');
  }

  // Why is Numa quiet? Surface the first unmet gate so it is obvious which
  // condition (session running / offers enabled / confirmed baseline) blocks
  // the automatic offers, plus the last policy reason.
  const gateKey =
    status !== 'on'
      ? 'focus.diag.cameraOff'
      : !hasStep
        ? 'focus.diag.noStep'
        : !running
          ? 'focus.diag.notRunning'
          : !offersEnabled
            ? 'focus.diag.offersOff'
            : !baseline?.confirmedOnTask
              ? baselineReadyNow
                ? 'focus.diag.baselineReady'
                : 'focus.diag.baseline'
              : 'focus.diag.ready';
  const gateText = t(gateKey, {
    seconds: baselineSeconds,
    valid: baselineValid,
    coverage: baselineCoveragePct,
  });
  const reasonText = t(`focus.reason.${decision?.reason || 'waiting'}`);

  // Turn the first unmet gate into a one-click action where the camera card can
  // perform it itself. "No active step" is resolved on the Movement tab, so it
  // stays a plain hint.
  const gateAction =
    gateKey === 'focus.diag.notRunning'
      ? { label: t('focus.session.start'), run: () => onResume?.() }
      : gateKey === 'focus.diag.offersOff'
        ? { label: t('focus.diag.enableOffers'), run: () => setOffersEnabled(true) }
        : gateKey === 'focus.diag.baselineReady'
          ? { label: t('focus.emotion.baselineConfirm'), run: confirmBaseline }
          : null;

  let cardMessage = '';
  if (def) {
    cardMessage = t(def.messageKey, {
      action: intervention.payload?.action || stepLabel || '',
    });
    if (def.scenario === 14 && intervention.payload?.allDone) {
      cardMessage = t('focus.intervention.done.allDone');
    }
  }

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

      <div className="focus-settings">
        <label className="focus-option">
          <input
            type="checkbox"
            checked={motionEnabled}
            onChange={(e) => setMotionEnabled(e.target.checked)}
          />
          {t('focus.emotion.motionLabel')}
        </label>
        <label className="focus-option">
          <input
            type="checkbox"
            checked={soundEnabled}
            onChange={(e) => setSoundEnabled(e.target.checked)}
          />
          {t('focus.emotion.soundLabel')}
        </label>
        {soundEnabled && (
          <div className="focus-sound">
            <label className="focus-option focus-option-range">
              <span>{t('focus.emotion.soundVolume')}</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={soundVolume}
                onChange={(e) => setSoundVolume(Number(e.target.value))}
              />
            </label>
            <div className="focus-sound-preview">
              <button
                type="button"
                className="focus-btn ghost"
                onClick={() => previewFocusSound('tu', soundVolume)}
              >
                {t('focus.emotion.previewTu')}
              </button>
              <button
                type="button"
                className="focus-btn ghost"
                onClick={() => previewFocusSound('tu-du', soundVolume)}
              >
                {t('focus.emotion.previewTuDu')}
              </button>
            </div>
          </div>
        )}
      </div>

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
                : `${baselineValid} · ${baselineCoveragePct}%`}
            </span>
          </div>
          {!baseline?.confirmedOnTask && (
            <>
              <p className="focus-hint">
                {baselineReadyNow
                  ? t('focus.emotion.baselineReady')
                  : t('focus.emotion.baselineCollecting', {
                      valid: baselineValid,
                      seconds: baselineSeconds,
                      coverage: baselineCoveragePct,
                    })}
              </p>
              <button
                type="button"
                className="focus-btn ghost"
                onClick={confirmBaseline}
                disabled={!baselineReadyNow}
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
        className={`focus-numa ${def ? 'offer' : 'idle'} ${showUnavailable ? 'unavailable' : ''}`}
        aria-live="polite"
      >
        <div className="focus-numa-visual">
          <NumaCharacter gesture={gesture} gestureKey={gestureKey} animate={motionEnabled} />
        </div>
        <div className="focus-numa-body">
          <p className="focus-numa-label">
            {numaNearby ? t('focus.emotion.numa') : t('focus.emotion.core')}
          </p>
          {showUnavailable ? (
            <p className="focus-numa-idle">{t('focus.intervention.unavailable.message')}</p>
          ) : def ? (
            <>
              <p className="focus-numa-message">{cardMessage}</p>
              {def.scenario === 12 && steps.length > 0 && (
                <ol className="focus-numa-steps">
                  {steps.map((step, index) => (
                    <li key={step.id || index}>{step.text ? step.text : t(step.key)}</li>
                  ))}
                </ol>
              )}
              {def.buttons.length > 0 && (
                <div className="focus-offer-actions">
                  {def.buttons.map(({ answer: value, label }) => (
                    <button
                      key={value}
                      type="button"
                      className={`focus-btn ${
                        value === def.buttons[0].answer ? 'primary' : 'ghost'
                      }`}
                      onClick={() => answer(value)}
                    >
                      {t(label)}
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <p className="focus-numa-idle">{statusText}</p>
              <p className="focus-numa-gate">
                {t('focus.diag.gate')}: {gateText} · {reasonText}
              </p>
              {gateAction && (
                <div className="focus-offer-actions">
                  <button type="button" className="focus-btn primary" onClick={gateAction.run}>
                    {gateAction.label}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {status === 'on' && !def && (
        <div className="focus-help-request">
          <button type="button" className="focus-btn ghost" onClick={requestHelp}>
            {t('focus.emotion.requestHelp')}
          </button>
        </div>
      )}

      {debug && (
        <div className="focus-debug">
          <p className="focus-numa-gate">
            {t('focus.diag.gate')}: {gateText} · {reasonText} · {baselineValid}/
            {baselineSeconds}s · {total}
          </p>
          {decision?.progress && (
            <p className="focus-numa-gate">
              {Object.entries(decision.progress)
                .map(([key, value]) => `${key} ${Number(value).toFixed(2)}`)
                .join(' · ')}
            </p>
          )}
          <div className="focus-debug-actions">
            {['check', 'help', 'support', 'point', 'pause', 'done'].map((id) => (
              <button
                key={id}
                type="button"
                className="focus-btn ghost"
                onClick={() => openIntervention(id, { action: stepLabel })}
              >
                {id}
              </button>
            ))}
          </div>
        </div>
      )}

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

      {answerLog.length > 0 && (
        <p className="focus-camera-note">
          {t('focus.emotion.answerCount', { count: answerLog.length })}
        </p>
      )}

      <p className="focus-camera-note">{t('focus.emotion.note')}</p>
    </div>
  );
};
