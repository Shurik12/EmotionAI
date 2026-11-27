import React from 'react';
import ReactDOM from 'react-dom/client';
import { MobileApp } from './MobileApp';

// The Capacitor WebView always opens `/index.html`, so NavigationContext would
// otherwise initialise on the landing's `home` page. Seed the default tab
// before the first render to avoid a flash of the wrong screen. MobileApp still
// normalises unknown pages on mount, so a failure here is harmless.
try {
  const { pathname } = window.location;
  if (pathname === '/' || pathname === '/index.html') {
    window.history.replaceState(null, '', '/detector');
  }
} catch {
  // Non-fatal.
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MobileApp />
  </React.StrictMode>,
);
