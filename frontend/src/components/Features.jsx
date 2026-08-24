import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Features = () => {
  const { t } = useLanguage();

  const featureCards = [
    t('feature_card1'),
    t('feature_card2'),
    t('feature_card3'),
    t('feature_card4'),
    t('feature_card5'),
    t('feature_card6'),
    t('feature_card7'),
  ];

  return (
    <section className="features-section">
      <div className="container">
        <h1>{t('features_title')}</h1>
        <h2 className="section-title">{t('features_main')}</h2>

        <div className="features-grid">
          {featureCards.map((text, index) => (
            <div key={index} className="feature-card">
              <p>{text}</p>
            </div>
          ))}
        </div>

        <div className="clients-block">
          <p>{t('clients_title')}</p>
          <p><strong>B2B:</strong> {t('clients_b2b')}</p>
          <p><strong>B2C / МСП:</strong> {t('clients_b2c')}</p>
        </div>
      </div>
    </section>
  );
};