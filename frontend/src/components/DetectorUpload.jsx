import React, { useRef, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const DetectorUpload = ({ 
  file, 
  fileName, 
  fileSize, 
  preview, 
  isProcessing,
  onFileSelect, 
  onClear 
}) => {
  const { t } = useLanguage();
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      onFileSelect(droppedFile);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
    e.target.value = '';
  };

  return (
    <div className="detector-upload">
      <div
        className={`dropzone ${isDragging ? 'is-dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-icon" aria-hidden="true">
          <svg viewBox="0 0 64 64">
            <path d="M20 44H14a10 10 0 0 1-1-19.95A16 16 0 0 1 43 20a12 12 0 0 1 2 23.83h-7" />
            <path d="M32 48V24M23 33l9-9 9 9" />
          </svg>
        </div>
        <strong>{t('detector.dragFile')}</strong>
        <span className="or">{t('common.or')}</span>

        <input
          ref={inputRef}
          type="file"
          className="file-input"
          accept="image/*,video/*,audio/*"
          onChange={handleFileChange}
          disabled={isProcessing}
        />

        <button
          type="button"
          className="btn choose-file"
          onClick={() => inputRef.current?.click()}
          disabled={isProcessing}
        >
          {t('detector.chooseFile')}
        </button>

        <p className="formats">{t('detector.supportedFormats')}</p>

        {fileName && (
          <div className="file-info">
            <span>{fileSize ? `${fileName} (${fileSize})` : fileName}</span>
            <button
              type="button"
              className="remove-file"
              onClick={onClear}
              disabled={isProcessing}
              aria-label={t('common.clear')}
            >
              ×
            </button>
          </div>
        )}
      </div>

      {preview?.type === 'audio' && (
        <div className="audio-preview">
          <audio controls src={preview.url} className="audio-player" />
        </div>
      )}
    </div>
  );
};
