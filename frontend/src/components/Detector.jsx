import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';
import { useFileUpload } from '../hooks/useFileUpload';
import { useProgress } from '../hooks/useProgress';
import { DetectorUpload } from './DetectorUpload';
import { DetectorResults } from './DetectorResults';
import { ProgressIndicator } from './ProgressIndicator';
import { ErrorMessage } from './ErrorMessage';
import { PROCESSING_MODES } from '../utils/constants';

export const Detector = () => {
  const { t } = useLanguage();
  const { navigateTo } = useNavigation();
  const [mode, setMode] = useState('burnout'); // Changed from 'audio_burnout' to 'burnout'
  const [consentGiven, setConsentGiven] = useState(false);
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
    uploadFile,
    setError: setUploadError,
  } = useFileUpload();

  const { progress, isComplete, startTracking, stopTracking } = useProgress();

  // Clear progress when results arrive
  useEffect(() => {
    if (results) {
      stopTracking();
      setIsProcessing(false);
    }
  }, [results, stopTracking]);

  const handleUpload = async () => {
    if (!consentGiven) {
      setError('errors.consentRequired');
      return;
    }

    setResults(null);
    setError(null);
    setIsProcessing(true);

    try {
      const data = await uploadFile(mode);
      
      startTracking(data.task_id, (result) => {
        setResults(result);
        setIsProcessing(false);
      });
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'errors.uploadFailed');
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

  return (
    <div className="detector-page">
      <section className="detector-hero">
        <div className="detector-hero-inner">
          <div className="detector-intro">
            <p className="detector-tagline">{t('detector.heroTagline')}</p>
            <h1>{t('detector.heroTitle')}</h1>
            <p className="detector-lead">{t('detector.heroLead')}</p>
          </div>

          <div className="detector-panel">
            <h2 className="detector-panel-title">{t('detector.panelInstruction')}</h2>

            <DetectorUpload
              file={file}
              fileName={fileName}
              fileSize={fileSize}
              preview={preview}
              isProcessing={isProcessing || isUploading}
              onFileSelect={handleFileSelect}
              onClear={handleClear}
            />

            <label className="field-label" htmlFor="mode-select">
              {t('detector.processingMode')}
            </label>
            <select
              id="mode-select"
              className="mode-select"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={isProcessing || isUploading}
            >
              {PROCESSING_MODES.map(({ value, labelKey }) => (
                <option key={value} value={value}>{t(labelKey)}</option>
              ))}
            </select>

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
              {isProcessing || isUploading ? t('detector.processing') : t('detector.analyze')}
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

      <section className="detector-benefits">
        <div className="detector-benefits-grid">
          <article className="detector-benefit">
            <div className="detector-benefit-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48">
                <path d="M27 3 12 26h10l-2 19 16-25H26l1-17Z" />
              </svg>
            </div>
            <div>
              <h3>{t('detector.benefits.fastTitle')}</h3>
              <p>{t('detector.benefits.fastText')}</p>
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
              <h3>{t('detector.benefits.modeTitle')}</h3>
              <p>{t('detector.benefits.modeText')}</p>
            </div>
          </article>

          <article className="detector-benefit">
            <div className="detector-benefit-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48">
                <path d="M8 40V25h7v15H8Zm13 0V14h7v26h-7Zm13 0V6h7v34h-7Z" />
              </svg>
            </div>
            <div>
              <h3>{t('detector.benefits.resultTitle')}</h3>
              <p>{t('detector.benefits.resultText')}</p>
            </div>
          </article>
        </div>
      </section>

      {results && <DetectorResults results={results} />}
    </div>
  );
};
