import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const CookieConsent = () => {
  const { t } = useLanguage();
  const [showConsent, setShowConsent] = useState(false);

  useEffect(() => {
    const accepted = localStorage.getItem('cookiesAccepted');
    if (!accepted) {
      setShowConsent(true);
    }
  }, []);

  const acceptCookies = () => {
    localStorage.setItem('cookiesAccepted', 'true');
    setShowConsent(false);
  };

  if (!showConsent) return null;

  return (
    <div className="cookie-consent">
      <div className="cookie-content">
        <p>
          {t('cookies.text')}
          <a href="#privacy">{t('cookies.more')}</a>
        </p>
        <button 
          className="btn btn-primary" 
          onClick={acceptCookies}
        >
          {t('cookies.accept')}
        </button>
      </div>
    </div>
  );
};