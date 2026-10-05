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
    { section: 'security', label: t('nav.security') },
    { page: 'focus', label: t('nav.focus') },
  ];

  const handleSectionClick = (section) => (e) => {
    e.preventDefault();
    setShowMobileMenu(false);
    navigateToSection(section);
  };

  const handlePageClick = (page) => (e) => {
    e.preventDefault();
    setShowMobileMenu(false);
    navigateTo(page);
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
        aria-label="RÁZUMA — участник Сколково"
      >
        <span className="brand-mark" style={{color: '#062b50'}}>RÁZUMA</span>
        <span className="brand-divider" aria-hidden="true" />
        <img 
          src="/static/skolkovo.webp" 
          alt="Участник Сколково" 
          className="brand-skolkovo"
        />
      </a>

      <nav className={`main-nav ${showMobileMenu ? 'show' : ''}`}>
        {navItems.map(({ section, page, label }) => (
          <a
            key={section || page}
            href={page ? `/${page}` : `#${section}`}
            className={`nav-link ${page && currentPage === page ? 'active' : ''}`}
            onClick={page ? handlePageClick(page) : handleSectionClick(section)}
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
            navigateToSection('contact');
          }}
        >
          {t('nav.contactUs')}
        </button>
      </div>
    </header>
  );
};
