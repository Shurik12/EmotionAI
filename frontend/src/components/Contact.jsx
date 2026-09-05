import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Contact = () => {
  const { t } = useLanguage();

  return (
    <div className="contact-page">
      <div className="container">
        <h1>{t('contact.title')}</h1>
        
        <div className="contact-section">
          <h2>{t('contact.legalInfo')}</h2>
          <div className="contact-info">
            <p>
              <strong>{t('contact.companyNameLabel')}: </strong> {t('contact.companyName')}
            </p>
            <p>
              <strong>{t('contact.legalAddressLabel')}: </strong> {t('contact.legalAddress')}
            </p>
            <p>
              <strong>{t('contact.inlnLabel')}: </strong> {t('contact.inln')}
            </p>
            <p>
              <strong>{t('contact.ogrnipLabel')}: </strong> {t('contact.ogrnip')}
            </p>
          </div>
        </div>
        
        <div className="contact-section">
          <h2>{t('contact.contactDetails')}</h2>
          <div className="contact-info">
            <p>
              <strong>{t('contact.emailLabel')}: </strong> 
              <a href={`mailto:${t('contact.email')}`}>{t('contact.email')}</a>
            </p>
            <p>
              <strong>{t('contact.phoneLabel')}: </strong> 
              <a href={`tel:${t('contact.phone').replace(/\s/g, '')}`}>{t('contact.phone')}</a>
            </p>
            <p>
              <strong>{t('contact.workingHoursLabel')}: </strong> {t('contact.workingHours')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};