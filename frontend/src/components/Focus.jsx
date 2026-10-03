import React, { useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Focus = () => {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState('movement');

  const tabs = [
    { id: 'movement', label: t('focus.tabs.movement') },
    { id: 'session', label: t('focus.tabs.session') },
    { id: 'how', label: t('focus.tabs.how') },
  ];

  const panels = {
    movement: { title: t('focus.movement.title'), text: t('focus.movement.text') },
    session: { title: t('focus.session.title'), text: t('focus.session.text') },
    how: { title: t('focus.how.title'), text: t('focus.how.text') },
  };

  const activePanel = panels[activeTab];

  return (
    <div className="focus-page">
      <section className="focus-hero">
        <span className="focus-badge">{t('focus.badge')}</span>
        <p className="focus-kicker">{t('focus.kicker')}</p>
        <h1 className="focus-title">{t('focus.title')}</h1>
        <p className="focus-subtitle">{t('focus.subtitle')}</p>
      </section>

      <div className="focus-tabs" role="tablist">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={activeTab === id}
            className={`focus-tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="focus-panel" role="tabpanel">
        <div className="focus-card">
          <h2>{activePanel.title}</h2>
          <p>{activePanel.text}</p>
        </div>
      </div>

      <p className="focus-privacy">{t('focus.privacy')}</p>
    </div>
  );
};
