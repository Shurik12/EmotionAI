import React, { useState, useEffect } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';

export const CookieConsent = () => {
  const { t } = useLanguage();
  const { navigateTo } = useNavigation();
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
          <a href="#privacy" onClick={(e) => { e.preventDefault(); navigateTo('privacy'); }}>{t('cookies.more')}</a>
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