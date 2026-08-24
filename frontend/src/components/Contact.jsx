import React from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Contact = () => {
  const { t } = useLanguage();

  // Use translations for both labels and values
  const legalInfo = [
    { label: t('contact.companyNameLabel'), value: t('contact.companyName') },
    { label: t('contact.legalAddressLabel'), value: t('contact.legalAddress') },
    { label: t('contact.innLabel'), value: t('contact.inn') },
    { label: t('contact.ogrnipLabel'), value: t('contact.ogrnip') },
  ];

  const contactInfo = [
    { label: t('contact.emailLabel'), value: t('contact.email') },
    { label: t('contact.phoneLabel'), value: t('contact.phone') },
    { label: t('contact.workingHoursLabel'), value: t('contact.workingHours') },
  ];

  return (
    <div className="contact-container">
      <div className="container">
        <section className="contact-hero">
          <h1>{t('contact.title')}</h1>
        </section>

        <section className="contact-content">
          <div className="contact-card">
            <h2>{t('contact.legalInfo')}</h2>
            <div className="info-grid">
              {legalInfo.map((item, index) => (
                <div key={index} className="info-row">
                  <strong>{item.label}:</strong>
                  <span>{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="contact-card">
            <h2>{t('contact.contactDetails')}</h2>
            <div className="info-grid">
              {contactInfo.map((item, index) => (
                <div key={index} className="info-row">
                  <strong>{item.label}:</strong>
                  <span>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};