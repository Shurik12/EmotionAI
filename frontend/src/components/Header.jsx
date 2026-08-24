import React, { useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

export const Header = ({ navigateTo, language, setLanguage }) => {
  const { t } = useLanguage();  // Make sure this is called correctly
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const handleNavClick = (path) => {
    navigateTo(path);
    setShowMobileMenu(false);
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const navItems = [
    { path: 'features', label: t('nav.features') },
    { path: 'detector', label: t('nav.demo') },
    { path: 'privacy', label: t('nav.privacy') },
    { path: 'contact', label: t('nav.contacts') },
  ];

  return (
    <header className="header">
      <button 
        className="mobile-menu-btn" 
        onClick={() => setShowMobileMenu(!showMobileMenu)}
        aria-label="Toggle menu"
      >
        ☰
      </button>

      <a 
        href="/" 
        className="logo" 
        onClick={(e) => {
          e.preventDefault();
          handleNavClick('home');
        }}
      >
        <img 
          src="/static/Razuma_Black.svg" 
          alt="Razuma Logo" 
          className="logo-icon"
        />
        <span>Razuma</span>
      </a>

      <nav className={`main-nav ${showMobileMenu ? 'show' : ''}`}>
        {navItems.map(({ path, label }) => (
          <a
            key={path}
            href={`#${path}`}
            className="nav-link"
            onClick={(e) => {
              e.preventDefault();
              handleNavClick(path);
            }}
          >
            {label}
          </a>
        ))}
      </nav>

      <div className="header-actions">
        <select 
          className="language-select" 
          value={language} 
          onChange={handleLanguageChange}
        >
          <option value="ru">Русский</option>
          <option value="en">English</option>
        </select>

        <button 
          className="btn btn-primary" 
          onClick={() => handleNavClick('detector')}
        >
          {t('nav.tryDemo')}
        </button>
      </div>
    </header>
  );
};