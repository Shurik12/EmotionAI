import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { apiClient } from '../api/client';

const FRAME_QUALITY = 0.85;
const MAX_STRIP_SAMPLES = 40;
// The server scores a frame in ~0.25 s, so anything from 1 s up is safe; the
// in-flight guard below makes a slow frame simply delay the next one.
const SAMPLE_INTERVAL_OPTIONS = [1, 3, 10];
const DEFAULT_SAMPLE_INTERVAL_SEC = 3;
const SESSION_POLL_MS = 1000;

// Camera + emotional dynamics for the Focus session (EMO-17, phase 4).
//
// The camera only starts after an explicit consent checkbox, stops as soon as
// the tab is hidden, and never records. While it is on, frames are streamed
// into a server-side session (POST /api/focus/session/<id>/frame) and the
// accumulated dynamics are polled back (GET /api/focus/session/<id>). The
// server does the scoring and the reaction classification; nothing is stored.
export const FocusCamera = ({ numaNearby = true }) => {
  const { t } = useLanguage();

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const requestRef = useRef(0);
  const aliveRef = useRef(true);
  const sessionIdRef = useRef(null);
  const updatedRef = useRef(0);
  const inFlightRef = useRef(false);
  const frameUrlRef = useRef(null);

  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState('off'); // off | asking | on | error
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [intervalSec, setIntervalSec] = useState(DEFAULT_SAMPLE_INTERVAL_SEC);
  const [message, setMessage] = useState('');
  const [frame, setFrame] = useState(null); // manual snapshot: { url, width, height }
  const [measuring, setMeasuring] = useState(false);
  const [reaction, setReaction] = useState(null); // latest server sample
  const [samples, setSamples] = useState([]); // [{ level, arousal }]
  const [total, setTotal] = useState(0); // samples measured this session

  const intervalMs = intervalSec * 1000;

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
      setSamples([]);
      setReaction(null);
      setTotal(0);
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
    setStatus('off');
    endSession();
  }, [endSession]);

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

  // Open the session and read the accumulated dynamics back.
  useEffect(() => {
    if (status !== 'on') return undefined;

    if (!sessionIdRef.current) startSession();

    const pollId = setInterval(async () => {
      const id = sessionIdRef.current;
      if (!id) return;
      try {
        const session = await apiClient.getFocusSession(id);
        if (!aliveRef.current) return;
        setSamples((session.samples || []).map((s) => ({ level: s.level, activation: s.activation })));
        setReaction(session.latest || null);
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

  const reactionBase = reaction?.level ? `focus.reaction.${reaction.level}` : null;
  const strip = samples.slice(-MAX_STRIP_SAMPLES);

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

      <div className={`focus-numa ${reaction?.level || 'idle'}`}>
        <p className="focus-numa-label">{numaNearby ? t('focus.reaction.numa') : t('focus.reaction.core')}</p>
        {reactionBase ? (
          <>
            <h4>{t(`${reactionBase}.title`)}</h4>
            <p>{t(`${reactionBase}.body`)}</p>
          </>
        ) : (
          <p className="focus-numa-idle">{t('focus.reaction.idle')}</p>
        )}
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
                key={`${index}-${sample.level}`}
                className={`focus-dynamics-bar ${sample.level}`}
                style={{ height: `${15 + (sample.activation ?? 0.5) * 85}%` }}
              />
            ))}
          </div>
        ) : (
          <p className="focus-hint">{t('focus.camera.dynamicEmpty')}</p>
        )}
      </div>

      <p className="focus-camera-note">{t('focus.camera.engineNote')}</p>
    </div>
  );
};
