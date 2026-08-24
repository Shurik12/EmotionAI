import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useFileUpload } from '../hooks/useFileUpload';
import { useProgress } from '../hooks/useProgress';
import { DetectorUpload } from './DetectorUpload';
import { DetectorResults } from './DetectorResults';
import { ProgressIndicator } from './ProgressIndicator';
import { ErrorMessage } from './ErrorMessage';
import { PROCESSING_MODES } from '../utils/constants';

export const Detector = () => {
  const { t } = useLanguage();
  const [mode, setMode] = useState('standard');
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
    <div className="detector-container">
      <h1 className="detector-title">{t('detector.title')}</h1>

      <DetectorUpload
        file={file}
        fileName={fileName}
        fileSize={fileSize}
        preview={preview}
        isProcessing={isProcessing || isUploading}
        onFileSelect={handleFileSelect}
        onClear={handleClear}
      />

      <div className="detector-options">
        <div className="option-group">
          <label htmlFor="mode-select">{t('detector.processingMode')}</label>
          <select
            id="mode-select"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            disabled={isProcessing || isUploading}
          >
            {PROCESSING_MODES.map(({ value, label }) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>

        <div className="option-group consent">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={consentGiven}
              onChange={(e) => setConsentGiven(e.target.checked)}
              disabled={isProcessing || isUploading}
            />
            <span>
              {t('detector.consent')} 
              <a href="#privacy"> {t('nav.privacy')}</a>
            </span>
          </label>
        </div>

        <button
          className="btn btn-primary submit-btn"
          onClick={handleUpload}
          disabled={!file || !consentGiven || isProcessing || isUploading}
        >
          {isProcessing || isUploading ? t('detector.processing') : t('detector.analyze')}
        </button>
      </div>

      {(error || uploadError) && (
        <ErrorMessage message={t(error || uploadError)} />
      )}
      
      {showProgress && (
        <ProgressIndicator 
          progress={progress} 
          isComplete={false}
        />
      )}

      {results && <DetectorResults results={results} />}
    </div>
  );
};