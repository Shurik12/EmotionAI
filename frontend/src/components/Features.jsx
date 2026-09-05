import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Features = () => {
  const { t } = useLanguage();

  // Get feature cards from translations
  const featureCards = t('features.cards', [], true); // Pass true to get array

  return (
    <section className="features-section">
      <div className="container">
        <h1>{t('features.title')}</h1>
        <h2 className="section-title">{t('features.main')}</h2>

        <div className="features-grid">
          {featureCards.map((text, index) => (
            <div key={index} className="feature-card">
              <p>{text}</p>
            </div>
          ))}
        </div>

        <div className="clients-block">
          <p>{t('features.clients.title')}</p>
          <p><strong>B2B:</strong> {t('features.clients.b2b')}</p>
          <p><strong>B2C / МСП:</strong> {t('features.clients.b2c')}</p>
        </div>
      </div>
    </section>
  );
};