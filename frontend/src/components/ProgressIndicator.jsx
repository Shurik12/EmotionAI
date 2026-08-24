import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const ProgressIndicator = ({ progress, isComplete }) => {
  const { t } = useLanguage();
  
  // Don't show progress if complete
  if (isComplete || !progress) return null;

  // Get the text, handling both string keys and direct text
  let text = progress.text;
  
  // If it's a translation key (starts with a letter and contains dots or is a known key)
  if (typeof text === 'string' && !text.includes(' ')) {
    // Try to translate it
    const translated = t(text);
    // If translation returns the same string (key not found), use the original
    text = translated !== text ? translated : text;
  }

  return (
    <div className="progress-container">
      <div className="progress-header">
        <h4>{t('common.processing')}</h4>
      </div>
      
      <div className="progress-wheel" />
      
      <div className="progress-text">{text}</div>
      
      {progress.value > 0 && (
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{ width: `${Math.min(progress.value * 100, 100)}%` }}
          />
        </div>
      )}
    </div>
  );
};