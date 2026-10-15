import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';
import { useFileUpload } from '../hooks/useFileUpload';
import { useProgress } from '../hooks/useProgress';
import { DetectorUpload } from './DetectorUpload';
import { ProgressIndicator } from './ProgressIndicator';
import { ErrorMessage } from './ErrorMessage';
import { apiClient } from '../api/client';
import '../styles/components/detector.css';
import '../styles/components/review.css';

// ── Colour helpers ──────────────────────────────────────────────
const VERDICT_COLORS = {
  'review.verdict.match': '#28a745',
  'review.verdict.minor': '#ffc107',
  'review.verdict.major': '#dc3545',
};

const getVerdictColor = (v) => VERDICT_COLORS[v] || '#6c757d';

// ── Score bar (1-10) ────────────────────────────────────────────
const ScoreBar = ({ label, value, color }) => {
  const pct = ((value - 1) / 9) * 100;
  return (
    <div className="emotion-item">
      <div className="emotion-label">
        <span>{label}</span>
        <span>{value.toFixed(1)}</span>
      </div>
      <div className="emotion-bar">
        <div
          className="emotion-fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
};

// ── Review component ────────────────────────────────────────────
export const Review = () => {
  const { t } = useLanguage();
  const { navigateTo } = useNavigation();
  const [consentGiven, setConsentGiven] = useState(false);
  const [selfScore, setSelfScore] = useState(7);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const {
    file,
    fileName,
    fileSize,
    preview,
    error: uploadError,
    isProcessing: isUploading,
    handleFileSelect,
    clearFile,
  } = useFileUpload();

  const { progress, isComplete, startTracking, stopTracking } = useProgress();

  // Stop polling if the page is left while a task is still running.
  useEffect(() => () => stopTracking(), [stopTracking]);

  const handleUpload = async () => {
    if (!consentGiven) {
      setError('errors.consentRequired');
      return;
    }

    setResults(null);
    setError(null);
    setIsProcessing(true);

    try {
      const data = await apiClient.uploadReview(file, selfScore);
      startTracking(data.task_id, (result) => {
        setResults(result);
        // The task can complete without a review (e.g. no face in the frame);
        // surface that instead of leaving an empty page.
        if (!result.review) {
          setError(result.error ? 'errors.processingFailed' : 'review.noEmotions');
        }
        setIsProcessing(false);
      });
    } catch (err) {
      console.error('Review upload error:', err);
      // Never show raw server text: map to translation keys.
      setError(/file type|supports image and video/i.test(err.message || '')
        ? 'errors.unsupportedFormat'
        : 'errors.uploadFailed');
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    clearFile();
    setResults(null);
    setError(null);
    setIsProcessing(false);
    stopTracking();
  };

  const showProgress = !isComplete && progress && !results;
  const review = results?.review;
  const isReviewReady = results?.complete && review;

  return (
    <div className="detector-page">
      <section className="detector-hero">
        <div className="detector-hero-inner">
          <div className="detector-intro">
            <p className="detector-tagline">{t('review.tagline')}</p>
            <h1>{t('review.title')}</h1>
            <p className="detector-lead">{t('review.lead')}</p>
          </div>

          <div className="detector-panel">
            <h2 className="detector-panel-title">{t('review.panelInstruction')}</h2>

            <DetectorUpload
              file={file}
              fileName={fileName}
              fileSize={fileSize}
              preview={preview}
              isProcessing={isProcessing || isUploading}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
              accept="image/*,video/*"
              formatsText={t('review.supportedFormats')}
            />

            {/* Self score */}
            <label className="field-label" htmlFor="self-score">
              {t('review.selfScoreLabel')}
            </label>
            <div className="score-input-row">
              <input
                id="self-score"
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={selfScore}
                onChange={(e) => setSelfScore(parseFloat(e.target.value))}
                disabled={isProcessing || isUploading}
                className="score-slider"
              />
              <span className="score-value-badge">{selfScore.toFixed(1)}</span>
            </div>
            <div className="score-hints">
              <span>{t('review.bad')}</span>
              <span>{t('review.excellent')}</span>
            </div>

            {/* Consent */}
            <label className="consent-row">
              <input
                type="checkbox"
                checked={consentGiven}
                onChange={(e) => setConsentGiven(e.target.checked)}
                disabled={isProcessing || isUploading}
              />
              <span>
                {t('detector.consent')} <a href="#privacy" onClick={(e) => { e.preventDefault(); navigateTo('privacy'); }}>{t('nav.privacy')}</a>
              </span>
            </label>

            <button
              className="btn analyze-btn"
              onClick={handleUpload}
              disabled={!file || !consentGiven || isProcessing || isUploading}
            >
              {isProcessing || isUploading ? t('review.processing') : t('review.analyze')}
            </button>

            {(error || uploadError) && (
              <ErrorMessage message={t(error || uploadError)} />
            )}

            {showProgress && (
              <ProgressIndicator
                progress={progress}
                isComplete={false}
              />
            )}
          </div>

          <img
            src="/static/demo.webp"
            alt="RAZUMA"
            className="detector-visual"
          />
        </div>
      </section>

      {/* Benefits strip — same as detector */}
      <section className="detector-benefits">
        <div className="detector-benefits-grid">
          <article className="detector-benefit">
            <div className="detector-benefit-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48">
                <path d="M27 3 12 26h10l-2 19 16-25H26l1-17Z" />
              </svg>
            </div>
            <div>
              <h3>{t('review.benefits.fastTitle')}</h3>
              <p>{t('review.benefits.fastText')}</p>
            </div>
          </article>

          <article className="detector-benefit">
            <div className="detector-benefit-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48">
                <circle cx="24" cy="24" r="7" />
                <path d="M24 5v5m0 28v5M5 24h5m28 0h5M10.6 10.6l3.6 3.6m19.6 19.6 3.6 3.6M37.4 10.6l-3.6 3.6M14.2 33.8l-3.6 3.6" />
              </svg>
            </div>
            <div>
              <h3>{t('review.benefits.modeTitle')}</h3>
              <p>{t('review.benefits.modeText')}</p>
            </div>
          </article>

          <article className="detector-benefit">
            <div className="detector-benefit-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48">
                <path d="M8 40V25h7v15H8Zm13 0V14h7v26h-7Zm13 0V6h7v34h-7Z" />
              </svg>
            </div>
            <div>
              <h3>{t('review.benefits.resultTitle')}</h3>
              <p>{t('review.benefits.resultText')}</p>
            </div>
          </article>
        </div>
      </section>

      {/* ── Results ─────────────────────────────────────────────── */}
      {isReviewReady && (
        <div className="results-container">
          <div className="result-card">
            <h2>{t('review.resultTitle')}</h2>

            <div className="emotion-results">
              <ScoreBar
                label={t('review.selfScore')}
                value={review.self_score}
                color="#2196F3"
              />
              <ScoreBar
                label={t('review.modelScore')}
                value={review.model_score}
                color="#E91E63"
              />
              <ScoreBar
                label={t('review.adjustedScore')}
                value={review.adjusted_score}
                color="#9C27B0"
              />
            </div>

            <div className="review-verdict">
              <span
                className="review-verdict-badge"
                style={{ backgroundColor: getVerdictColor(review.verdict) }}
              >
                {t(review.verdict)}
              </span>
              <span className="review-delta">
                {t('review.delta')}: {review.delta.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
