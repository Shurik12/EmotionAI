import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Footer = () => {
  const { t } = useLanguage();

  return (
    <footer className="footer" id="about">
      <div className="footer__legal">
        <div className="footer__copyright">
          {t('footer.copyright')}
        </div>

        <div className="footer__company">
          <strong>{t('contact.companyName')}</strong>
          <a href={`mailto:${t('contact.email')}`}>{t('contact.email')}</a>
          <span>{t('contact.ogrnipLabel')}: {t('contact.ogrnip')}</span>
          <span>{t('contact.inlnLabel')}: {t('contact.inln')}</span>
        </div>

        <address className="footer__address">
          {t('contact.legalAddress')}
        </address>
      </div>
    </footer>
  );
};
