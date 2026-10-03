import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { apiClient } from '../api/client';
import { buildCameraReaction } from '../utils/focusCamera';

const FRAME_QUALITY = 0.85;
const POLL_INTERVAL_MS = 1000;
const MAX_POLLS = 60;
const MAX_SAMPLES = 12;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Camera + emotional dynamics for the Focus session (EMO-17, phase 4).
//
// The camera only starts after an explicit consent checkbox, stops as soon as
// the tab is hidden, and never records: a single JPEG frame is captured on
// demand and sent to the existing Razuma core (POST /api/upload) only when the
// user asks for it. The returned signal is mapped locally to a calm next step.
export const FocusCamera = ({ numaNearby = true }) => {
  const { t } = useLanguage();

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const requestRef = useRef(0);
  const aliveRef = useRef(true);

  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState('off'); // off | asking | on | error
  const [message, setMessage] = useState('');
  const [frame, setFrame] = useState(null); // { blob, url, width, height }
  const [analysis, setAnalysis] = useState('idle'); // idle | processing | done | error
  const [progressValue, setProgressValue] = useState(0);
  const [reaction, setReaction] = useState(null);
  const [samples, setSamples] = useState([]);

  const stopCamera = useCallback(() => {
    requestRef.current += 1;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus('off');
  }, []);

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
    };
  }, [stopCamera]);

  const handleStreamEnded = useCallback(() => {
    streamRef.current = null;
    setStatus('error');
    setMessage(t('focus.camera.streamEnded'));
  }, [t]);

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

  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || video.readyState < 2) {
      setMessage(t('focus.camera.captureHint'));
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        if (frame?.url) URL.revokeObjectURL(frame.url);
        setFrame({ blob, url: URL.createObjectURL(blob), width: canvas.width, height: canvas.height });
        setMessage('');
      },
      'image/jpeg',
      FRAME_QUALITY,
    );
  };

  const analyzeFrame = async () => {
    if (!frame?.blob || analysis === 'processing') return;

    setAnalysis('processing');
    setProgressValue(0);
    setReaction(null);
    setMessage('');

    try {
      const file = new File([frame.blob], 'focus-frame.jpg', { type: 'image/jpeg' });
      const { task_id: taskId } = await apiClient.uploadFile(file, 'standard');

      let data = null;
      for (let attempt = 0; attempt < MAX_POLLS; attempt += 1) {
        await sleep(POLL_INTERVAL_MS);
        if (!aliveRef.current) return;
        data = await apiClient.getProgress(taskId);
        setProgressValue(data?.progress || 0);
        if (data?.complete) break;
      }

      if (!data?.complete) throw new Error('analysis timeout');
      if (!aliveRef.current) return;

      const next = buildCameraReaction(data.result ?? data);
      setReaction(next);
      setSamples((prev) =>
        [...prev, { arousal: next.signal?.arousal ?? 0, kind: next.kind }].slice(-MAX_SAMPLES),
      );
      setAnalysis('done');
    } catch (err) {
      if (!aliveRef.current) return;
      setAnalysis('error');
      setMessage(t('focus.camera.error'));
    }
  };

  const reactionBase = reaction?.baseKey;

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
        {frame
          ? t('focus.camera.frameReady', { width: frame.width, height: frame.height })
          : t('focus.camera.frameNone')}
      </p>

      {frame && (
        <figure className="focus-camera-snapshot">
          <img src={frame.url} alt={t('focus.camera.frameAlt')} />
        </figure>
      )}

      <label className="focus-option">
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => setConsent(e.target.checked)}
          disabled={status === 'on' || status === 'asking'}
        />
        {t('focus.camera.consent')}
      </label>

      <div className="focus-actions">
        {status === 'on' ? (
          <>
            <button type="button" className="focus-btn primary" onClick={captureFrame}>
              {t('focus.camera.capture')}
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
        <div className="focus-actions">
          <button
            type="button"
            className="focus-btn primary"
            onClick={analyzeFrame}
            disabled={analysis === 'processing'}
          >
            {analysis === 'processing' ? t('focus.camera.analyzing') : t('focus.camera.analyze')}
          </button>
        </div>
      )}

      {analysis === 'processing' && (
        <div className="focus-progress">
          <div className="focus-progress-bar" style={{ width: `${Math.max(5, progressValue)}%` }} />
        </div>
      )}

      <div className={`focus-numa ${reaction?.kind || 'idle'}`}>
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
        <p className="focus-dynamics-title">{t('focus.camera.dynamic')}</p>
        {samples.length ? (
          <div className="focus-dynamics-bars" aria-hidden="true">
            {samples.map((sample, index) => (
              <span
                key={`${index}-${sample.kind}`}
                className={`focus-dynamics-bar ${sample.kind}`}
                style={{ height: `${20 + sample.arousal * 80}%` }}
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
