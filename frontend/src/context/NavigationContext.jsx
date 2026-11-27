import React, { createContext, useState, useEffect } from 'react';

// Export the context directly
export const NavigationContext = createContext();

// Read the current route from the URL. Used both for the initial state and on
// popstate, so a deep link (`/focus`, `/detector`, ...) renders correctly on the
// first paint instead of flashing `home` while an effect catches up.
const readPath = () => {
  if (typeof window === 'undefined') return 'home';
  return window.location.pathname.replace(/^\//, '') || 'home';
};

export const NavigationProvider = ({ children }) => {
  const [currentPage, setCurrentPage] = useState(readPath);

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPage(readPath());
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigateTo = (path) => {
    const cleanPath = path.replace(/^\//, '');
    setCurrentPage(cleanPath);
    
    const url = cleanPath === 'home' ? '/' : `/${cleanPath}`;
    window.history.pushState(null, '', url);
    
    const titles = {
      home: 'Razuma | Emotion Recognition',
      features: 'Razuma | Features',
      detector: 'Razuma | Demo',
      focus: 'Razuma | Focus',
      checkout: 'Razuma | Checkout',
      privacy: 'Razuma | Privacy',
      contact: 'Razuma | Contact',
    };
    document.title = titles[cleanPath] || 'Razuma';
  };

  const navigateToSection = (sectionId) => {
    if (currentPage !== 'home') {
      navigateTo('home');
    }
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  };

  const value = {
    currentPage,
    navigateTo,
    navigateToSection,
  };

  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
};