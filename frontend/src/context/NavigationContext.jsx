import React, { createContext, useState, useEffect } from 'react';

// Export the context directly
export const NavigationContext = createContext();

export const NavigationProvider = ({ children }) => {
  const [currentPage, setCurrentPage] = useState('home');

  useEffect(() => {
    const path = window.location.pathname.replace(/^\//, '');
    setCurrentPage(path || 'home');

    const handlePopState = () => {
      const newPath = window.location.pathname.replace(/^\//, '');
      setCurrentPage(newPath || 'home');
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
      privacy: 'Razuma | Privacy',
      contact: 'Razuma | Contact',
    };
    document.title = titles[cleanPath] || 'Razuma';
  };

  const value = {
    currentPage,
    navigateTo,
  };

  return (
    <NavigationContext.Provider value={value}>
      {children}
    </NavigationContext.Provider>
  );
};