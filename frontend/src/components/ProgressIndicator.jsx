import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const ProgressIndicator = ({ progress, isComplete }) => {
  const { t } = useLanguage();
  
  // Don't show progress if complete
  if (isComplete || !progress) return null;

  // Get the text, handling both translation keys and direct text.
  // t() returns the key unchanged when no translation is found, so it's
  // always safe to call — non-key strings pass through verbatim.
  let text = typeof progress.text === 'string' ? t(progress.text) : progress.text;

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