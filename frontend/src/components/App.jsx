import React, { useState } from 'react';  // Make sure React and useState are imported
import { LanguageProvider } from '../context/LanguageContext';
import { NavigationProvider } from '../context/NavigationContext';
import { Header } from './Header';
import { Footer } from './Footer';
import { Home } from './Home';
import { Features } from './Features';
import { Detector } from './Detector';
import { Privacy } from './Privacy';
import { Contact } from './Contact';
import { CookieConsent } from './CookieConsent';
import { ApplicationModal } from './ApplicationModal';
import { useNavigation } from '../hooks/useNavigation';
import { useLanguage } from '../hooks/useLanguage';
import '../styles/global.css';
import '../styles/components.css';

// Main app content with hooks
const AppContent = () => {
  const { currentPage, navigateTo } = useNavigation();
  const { language, setLanguage } = useLanguage();
  const [showModal, setShowModal] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState('');

  const openModal = (plan) => {
    setSelectedPlan(plan);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
  };

  const renderContent = () => {
    const props = { navigateTo, openModal };
    
    switch (currentPage) {
      case 'home':
        return <Home {...props} />;
      case 'features':
        return <Features />;
      case 'detector':
        return <Detector />;
      case 'privacy':
        return <Privacy />;
      case 'contact':
        return <Contact />;
      default:
        return <Home {...props} />;
    }
  };

  return (
    <div className="app">
      <Header 
        navigateTo={navigateTo} 
        language={language} 
        setLanguage={setLanguage} 
      />
      
      <main className="main-content">
        {renderContent()}
      </main>
      
      <Footer navigateTo={navigateTo} />
      <CookieConsent />
      
      {showModal && (
        <ApplicationModal 
          selectedPlan={selectedPlan} 
          closeModal={closeModal} 
        />
      )}
    </div>
  );
};

// Root App with providers
export const App = () => {
  return (
    <LanguageProvider>
      <NavigationProvider>
        <AppContent />
      </NavigationProvider>
    </LanguageProvider>
  );
};