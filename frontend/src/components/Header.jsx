import React, { useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { useNavigation } from '../hooks/useNavigation';

export const Header = ({ language, setLanguage }) => {
  const { t } = useLanguage();
  const { currentPage, navigateTo, navigateToSection } = useNavigation();
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const navItems = [
    { section: 'solutions', label: t('nav.solutions') },
    { section: 'industries', label: t('nav.industries') },
    { section: 'technology', label: t('nav.technology') },
    { section: 'cases', label: t('nav.cases') },
    { section: 'about', label: t('nav.about') },
  ];

  const handleSectionClick = (section) => (e) => {
    e.preventDefault();
    setShowMobileMenu(false);
    navigateToSection(section);
  };

  const handleLogoClick = (e) => {
    e.preventDefault();
    setShowMobileMenu(false);
    if (currentPage === 'home') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      navigateTo('home');
    }
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

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
        className="brand-lockup" 
        onClick={handleLogoClick}
        aria-label="RAZUMA — участник Сколково"
      >
        <img 
          src="/static/razuma.svg" 
          alt="" 
          className="brand-mark"
        />
        <span className="brand-divider" aria-hidden="true" />
        <img 
          src="/static/skolkovo.webp" 
          alt="Участник Сколково" 
          className="brand-skolkovo"
        />
      </a>

      <nav className={`main-nav ${showMobileMenu ? 'show' : ''}`}>
        {navItems.map(({ section, label }) => (
          <a
            key={section}
            href={`#${section}`}
            className="nav-link"
            onClick={handleSectionClick(section)}
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
          className="header-cta" 
          onClick={() => {
            setShowMobileMenu(false);
            navigateTo('contact');
          }}
        >
          {t('nav.contactUs')}
        </button>
      </div>
    </header>
  );
};
