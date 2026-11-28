import React, { useEffect, useState } from 'react';
import { LanguageProvider } from '../context/LanguageContext';
import { NavigationProvider } from '../context/NavigationContext';
import { useNavigation } from '../hooks/useNavigation';
import { useLanguage } from '../hooks/useLanguage';
import { Detector } from '../components/Detector';
import { Focus } from '../components/Focus';
import { Privacy } from '../components/Privacy';
import { apiClient } from '../api/client';
import '../styles/global.css';
import './mobile.css';

// The Capacitor shell renders only product screens — the marketing landing is
// intentionally not shipped inside the app. Tabs piggyback on the existing
// NavigationContext so every in-page `navigateTo('privacy')` (e.g. the consent
// link in Detector) keeps working.
const TABS = [
  { page: 'detector', labelKey: 'nav.demo' },
  { page: 'focus', labelKey: 'nav.focus' },
  { page: 'privacy', labelKey: 'nav.privacy' },
];

const PAGES = TABS.map((tab) => tab.page);

const ICONS = {
  detector: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="3" />
      <circle cx="12" cy="10" r="2.2" />
      <path d="M7.5 17c.9-2 2.6-3 4.5-3s3.6 1 4.5 3" />
    </>
  ),
  focus: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3.4" />
    </>
  ),
  privacy: (
    <>
      <path d="M12 3l7 3v6c0 4.2-2.9 7.4-7 9-4.1-1.6-7-4.8-7-9V6l7-3z" />
      <path d="M9.5 12l1.8 1.8L14.8 10" />
    </>
  ),
};

const TabIcon = ({ page }) => (
  <svg
    viewBox="0 0 24 24"
    aria-hidden="true"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.6"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {ICONS[page]}
  </svg>
);

const Shell = () => {
  const { currentPage, navigateTo } = useNavigation();
  const { t, language, setLanguage } = useLanguage();
  const [deleteState, setDeleteState] = useState('idle'); // idle | working | done | error

  const page = PAGES.includes(currentPage) ? currentPage : 'detector';

  // Landing-only routes (home/features/contact/checkout) have no mobile screen;
  // fold them into the detector tab.
  useEffect(() => {
    if (!PAGES.includes(currentPage)) navigateTo('detector');
  }, [currentPage, navigateTo]);

  // Data deletion: revoking the device token also drops the saved baseline on
  // the server, so this is the app's "delete my data" action.
  const handleDeleteData = async () => {
    if (deleteState === 'working' || deleteState === 'done') return;
    if (!window.confirm(t('mobile.deleteConfirm'))) return;

    setDeleteState('working');
    try {
      await apiClient.revokeDeviceIdentity();
      setDeleteState('done');
    } catch {
      setDeleteState('error');
    }
  };

  return (
    <div className="mobile-shell">
      <header className="mobile-appbar">
        <span className="mobile-appbar-title">Razuma</span>
        <div className="mobile-lang" role="group" aria-label="Language">
          <button
            type="button"
            className={language === 'ru' ? 'is-active' : ''}
            onClick={() => setLanguage('ru')}
          >
            RU
          </button>
          <button
            type="button"
            className={language === 'en' ? 'is-active' : ''}
            onClick={() => setLanguage('en')}
          >
            EN
          </button>
        </div>
      </header>

      {/* `key={page}` remounts the screen on tab switch, so a detector result
          never leaks into a fresh analysis and leaving Focus stops the camera. */}
      <main className="app mobile-content" key={page}>
        {page === 'detector' && <Detector />}
        {page === 'focus' && <Focus />}
        {page === 'privacy' && (
          <>
            <p className="mobile-wellness-note">{t('mobile.wellnessNote')}</p>
            <section className="mobile-data-actions">
              <h2>{t('mobile.dataTitle')}</h2>
              <p>{t('mobile.dataText')}</p>
              <button
                type="button"
                className="mobile-danger"
                onClick={handleDeleteData}
                disabled={deleteState === 'working' || deleteState === 'done'}
              >
                {deleteState === 'working'
                  ? t('mobile.deleting')
                  : deleteState === 'done'
                    ? t('mobile.deleted')
                    : t('mobile.delete')}
              </button>
              {deleteState === 'error' && (
                <p className="mobile-delete-error">{t('mobile.deleteError')}</p>
              )}
            </section>
            <Privacy />
          </>
        )}
      </main>

      <nav className="mobile-tabbar">
        {TABS.map((tab) => {
          const active = tab.page === page;
          return (
            <button
              key={tab.page}
              type="button"
              className={`mobile-tab${active ? ' is-active' : ''}`}
              aria-current={active ? 'page' : undefined}
              onClick={() => navigateTo(tab.page)}
            >
              <TabIcon page={tab.page} />
              <span>{t(tab.labelKey)}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
};

export const MobileApp = () => (
  <LanguageProvider>
    <NavigationProvider>
      <Shell />
    </NavigationProvider>
  </LanguageProvider>
);
