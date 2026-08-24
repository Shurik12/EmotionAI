import React, { useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Privacy = () => {
  const { t, language } = useLanguage();
  const [iframeError, setIframeError] = useState(false);

  return (
    <div className="privacy-container">
      <div className="container">
        <h1>{t('nav.privacy')}</h1>
        
        {!iframeError ? (
          <div className="privacy-iframe-container">
            <iframe
              src={`/static/privacy-policy.html?lang=${language}&t=${Date.now()}`}
              className="privacy-iframe"
              title={t('nav.privacy')}
              onError={() => setIframeError(true)}
            />
          </div>
        ) : (
          <div className="privacy-fallback">
            <p>{t('privacy_error')}</p>
          </div>
        )}
      </div>
    </div>
  );
};