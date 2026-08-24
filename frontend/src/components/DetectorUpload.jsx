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

  // Only ONE supported formats line
  const supportedFormats = 'JPG, PNG, MP4, AVI, WEBM, MP3, WAV, AAC, OGG, FLAC (макс. 50MB)';

  return (
    <div className="detector-upload">
      <div 
        className={`upload-zone ${isDragging ? 'dragging' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-icon">📁</div>
        <h3>{t('detector.dragFile')}</h3>
        <p className="or-text">{t('common.or')}</p>
        
        <input
          ref={inputRef}
          type="file"
          className="file-input"
          accept="image/*,video/*,audio/*"
          onChange={handleFileChange}
          disabled={isProcessing}
        />
        
        <button 
          className="btn btn-secondary"
          onClick={() => inputRef.current?.click()}
          disabled={isProcessing}
        >
          {t('detector.chooseFile')}
        </button>
        
        <small className="supported-formats">
          {supportedFormats}
        </small>
      </div>

      {fileName && (
        <div className="file-info">
          <span className="file-name">{fileName}</span>
          <span className="file-size">({fileSize})</span>
          <button 
            className="btn btn-clear"
            onClick={onClear}
            disabled={isProcessing}
          >
            {t('common.clear')}
          </button>
        </div>
      )}

      {preview?.type === 'audio' && (
        <div className="audio-preview">
          <audio controls src={preview.url} className="audio-player" />
        </div>
      )}
    </div>
  );
};