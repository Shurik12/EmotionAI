import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Footer = ({ navigateTo }) => {
  const { t } = useLanguage();

  const footerLinks = [
    { path: 'detector', label: t('footer.demo') },
    { path: 'privacy', label: t('footer.privacy') },
    { path: 'contact', label: t('footer.contacts') },
  ];

  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="footer-logo">Razuma</div>
        
        <div className="footer-links">
          {footerLinks.map(({ path, label }) => (
            <a
              key={path}
              href={`#${path}`}
              onClick={(e) => {
                e.preventDefault();
                navigateTo(path);
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