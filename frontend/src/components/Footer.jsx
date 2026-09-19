import React from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';

export const Footer = () => {
  const { t } = useLanguage();
  const { navigateToSection } = useNavigation();

  const footerLinks = [
    { section: 'solutions', label: t('nav.solutions') },
    { section: 'industries', label: t('nav.industries') },
    { section: 'technology', label: t('nav.technology') },
    { section: 'cases', label: t('nav.cases') },
    { section: 'about', label: t('nav.about') },
  ];

  return (
    <footer className="footer" id="about">
      <div className="footer-content">
        <div className="footer-logo">RAZUMA</div>
        <div className="footer-tagline">{t('footer.tagline')}</div>
        
        <div className="footer-links">
          {footerLinks.map(({ section, label }) => (
            <a
              key={section}
              href={`#${section}`}
              onClick={(e) => {
                e.preventDefault();
                navigateToSection(section);
              }}
            >
              {label}
            </a>
          ))}
        </div>
        
        <p className="footer-copyright">{t('footer.copyright')}</p>
      </div>
    </footer>
  );
};
