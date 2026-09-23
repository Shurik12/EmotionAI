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
  ];

  return (
    <footer className="footer" id="about">
      <div className="footer-content">
        <p className="footer-copyright">{t('footer.copyright')}</p>

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
      </div>
    </footer>
  );
};
