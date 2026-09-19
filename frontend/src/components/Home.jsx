import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
};

const STEP_ICONS = [
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <rect x="3" y="7" width="12" height="10" rx="2" />
    <path d="M15 10.5 21 7v10l-6-3.5z" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <rect x="7" y="7" width="10" height="10" rx="2" />
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M5 20v-8M12 20V6M19 20v-6" />
    <path d="M3 20h18" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M4 13l5 5L20 6" />
  </svg>,
];

const ANALYSIS_ICONS = [
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <rect x="3" y="7" width="12" height="10" rx="2" />
    <path d="M15 10.5 21 7v10l-6-3.5z" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <rect x="9" y="3" width="6" height="10" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <circle cx="9.5" cy="12" r="4.5" />
    <circle cx="14.5" cy="12" r="4.5" />
  </svg>,
];

const BENEFIT_ICONS = [
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M12 3 3 8l9 5 9-5-9-5z" />
    <path d="M3 13l9 5 9-5" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M3 17l6-6 4 4 8-8" />
    <path d="M17 7h4v4" />
  </svg>,
  <svg viewBox="0 0 24 24" aria-hidden="true" {...stroke}>
    <path d="M9 3v4M15 3v4M7 7h10v5a5 5 0 0 1-10 0V7zM12 17v4" />
  </svg>,
];

const INDUSTRY_META = [
  { anchor: 'hr', num: '01', image: '/static/hr.png' },
  { anchor: 'industry', num: '02', image: '/static/industry.png' },
  { anchor: 'bank', num: '03', image: '/static/bank.png' },
  { anchor: 'research', num: '04', image: '/static/research.png' },
];

export const Home = () => {
  const { t } = useLanguage();
  const { navigateTo, navigateToSection } = useNavigation();

  const steps = t('landing.how.steps');
  const analysisCards = t('landing.analysis.cards');
  const industryCards = t('landing.industries.cards').map((card, index) => ({
    ...card,
    ...INDUSTRY_META[index],
  }));
  const solutionCards = t('landing.solutions.cards');

  const handleSectionClick = (section) => (e) => {
    e.preventDefault();
    navigateToSection(section);
  };

  const handleContactClick = (e) => {
    e.preventDefault();
    navigateTo('contact');
  };

  return (
    <div className="landing-page">
      <section className="landing-hero">
        <div className="landing-hero-media" aria-hidden="true" />
        <div className="landing-container landing-hero-grid">
          <div className="landing-hero-copy">
            <p className="landing-eyebrow">{t('landing.eyebrow')}</p>
            <h1>{t('landing.heroTitle')}</h1>
            <p className="landing-hero-lead">{t('landing.heroLead')}</p>
            <p className="landing-hero-sub">{t('landing.heroSub')}</p>

            <div className="landing-hero-actions">
              <button
                type="button"
                className="landing-btn landing-btn-dark"
                onClick={() => navigateTo('detector')}
              >
                {t('landing.tryDemo')} <span aria-hidden="true">→</span>
              </button>
              <button
                type="button"
                className="landing-btn landing-btn-light"
                onClick={() => navigateTo('contact')}
              >
                {t('landing.discussPilot')}
              </button>
            </div>

            <div className="landing-hero-links" aria-label={t('landing.industriesLabel')}>
              {industryCards.map((card, index) => (
                <React.Fragment key={card.anchor}>
                  {index > 0 && <span aria-hidden="true">•</span>}
                  <a href={`#${card.anchor}`} onClick={handleSectionClick(card.anchor)}>
                    {card.title}
                  </a>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="landing-section" id="technology">
        <div className="landing-container">
          <div className="landing-section-head">
            <div>
              <h2>{t('landing.how.title')}</h2>
              <p>{t('landing.how.subtitle')}</p>
            </div>
            <a className="landing-text-link" href="#analysis" onClick={handleSectionClick('analysis')}>
              {t('landing.how.link')} →
            </a>
          </div>

          <div className="landing-steps">
            {steps.map((step, index) => (
              <React.Fragment key={step.title}>
                {index > 0 && (
                  <div className="landing-arrow" aria-hidden="true">
                    ›
                  </div>
                )}
                <article className="landing-step">
                  <div className="landing-step-icon">{STEP_ICONS[index]}</div>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </article>
              </React.Fragment>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-tint" id="analysis">
        <div className="landing-container">
          <div className="landing-section-head">
            <div>
              <h2>{t('landing.analysis.title')}</h2>
              <p>{t('landing.analysis.subtitle')}</p>
            </div>
          </div>

          <div className="landing-analysis-grid">
            {analysisCards.map((card, index) => (
              <article className="landing-info-card" key={card.title}>
                <div className="landing-circle-icon">{ANALYSIS_ICONS[index]}</div>
                <div>
                  <h3>{card.title}</h3>
                  <ul>
                    {card.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section" id="industries">
        <div className="landing-container">
          <div className="landing-section-head">
            <div>
              <h2>{t('landing.industries.title')}</h2>
              <p>{t('landing.industries.subtitle')}</p>
            </div>
            <a className="landing-text-link" href="#contact" onClick={handleContactClick}>
              {t('landing.industries.link')} →
            </a>
          </div>

          <div className="landing-industry-grid">
            {industryCards.map((card) => (
              <article className="landing-industry-card" id={card.anchor} key={card.anchor}>
                <div className="landing-num">{card.num}</div>
                <h3>{card.title}</h3>
                <img src={card.image} alt={card.alt} loading="lazy" />
                <h4>{card.headline}</h4>
                <ul>
                  {card.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
                <a className="landing-card-link" href="#contact" onClick={handleContactClick}>
                  {t('landing.industries.more')} →
                </a>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-tint" id="solutions">
        <div className="landing-container">
          <div className="landing-section-head">
            <div>
              <h2>{t('landing.solutions.title')}</h2>
              <p>{t('landing.solutions.subtitle')}</p>
            </div>
          </div>

          <div className="landing-benefit-grid">
            {solutionCards.map((card, index) => (
              <article className="landing-benefit" key={card.title}>
                <div className="landing-circle-icon">{BENEFIT_ICONS[index]}</div>
                <h3>{card.title}</h3>
                <p>{card.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-decision" id="cases">
        <div className="landing-container landing-decision-grid">
          <div>
            <h2>{t('landing.decision.title')}</h2>
            <p>{t('landing.decision.text')}</p>
          </div>
          <div className="landing-decision-flow">
            <span>RAZUMA</span>
            <b aria-hidden="true">→</b>
            <span>{t('landing.decision.signal')}</span>
            <b aria-hidden="true">→</b>
            <span>{t('landing.decision.yours')}</span>
          </div>
        </div>
      </section>

      <section className="landing-cta" id="demo">
        <div className="landing-container landing-cta-grid">
          <div>
            <h2>{t('landing.cta.title')}</h2>
            <p>{t('landing.cta.text')}</p>
          </div>

          <div className="landing-cta-note">{t('landing.cta.note')}</div>

          <div>
            <h3>{t('landing.cta.demoTitle')}</h3>
            <p>{t('landing.cta.demoText')}</p>
            <button
              type="button"
              className="landing-btn landing-btn-dark"
              onClick={() => navigateTo('detector')}
            >
              {t('landing.cta.demoBtn')} <span aria-hidden="true">→</span>
            </button>
          </div>

          <div id="contact">
            <h3>{t('landing.cta.contactTitle')}</h3>
            <p>{t('landing.cta.contactText')}</p>
            <button
              type="button"
              className="landing-btn landing-btn-light"
              onClick={() => navigateTo('contact')}
            >
              {t('landing.cta.contactBtn')}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
