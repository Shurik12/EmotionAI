import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Home = ({ navigateTo, openModal }) => {
  const { t } = useLanguage();

  const pricingPlans = [
    {
      id: 'free',
      name: t('pricing.free.name'),
      price: t('pricing.free.price'),
      features: t('pricing.free.features'),
      buttonText: t('pricing.usePlan'),
      isFree: true,
    },
    {
      id: 'light',
      name: t('pricing.light.name'),
      price: t('pricing.light.price'),
      features: t('pricing.light.features'),
      buttonText: t('pricing.applyPlan'),
      isFree: false,
    },
    {
      id: 'pro',
      name: t('pricing.pro.name'),
      price: t('pricing.pro.price'),
      features: t('pricing.pro.features'),
      buttonText: t('pricing.applyPlan'),
      isFree: false,
    },
    {
      id: 'business',
      name: t('pricing.business.name'),
      price: t('pricing.business.price'),
      features: t('pricing.business.features'),
      buttonText: t('pricing.applyPlan'),
      isFree: false,
    },
  ];

  const homeFeatures = [
    t('home.features.promo'),
    t('home.features.ux'),
    t('home.features.experience'),
    t('home.features.presentations'),
    t('home.features.scripts'),
  ];

  return (
    <>
      <section className="hero">
        <div className="container">
          <h1>{t('home.title')}</h1>
          <p className="hero-description">{t('home.description')}</p>
          
          <h3 className="platform-usage">{t('home.platformUsage')}</h3>
          
          <div className="features-grid small">
            {homeFeatures.map((feature, index) => (
              <div key={index} className="feature-card small">
                <p>{feature}</p>
              </div>
            ))}
          </div>

          <div className="hero-actions">
            <button 
              className="btn btn-primary" 
              onClick={() => navigateTo('detector')}
            >
              {t('home.analyzeNow')}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={() => navigateTo('features')}
            >
              {t('home.learnMore')}
            </button>
          </div>
        </div>
      </section>

      <section className="pricing-section">
        <div className="container">
          <h2 className="section-title">{t('pricing.title')}</h2>
          
          <div className="pricing-grid">
            {pricingPlans.map((plan) => (
              <div key={plan.id} className="pricing-card">
                <h3>{plan.name}</h3>
                <div className="price">{plan.price}</div>
                <ul className="features-list">
                  {plan.features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
                </ul>
                <button 
                  className="btn btn-primary"
                  onClick={() => {
                    if (plan.isFree) {
                      navigateTo('detector');
                    } else {
                      openModal(plan.id);
                    }
                  }}
                >
                  {plan.buttonText}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
};