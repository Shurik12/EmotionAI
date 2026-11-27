import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'tech.razuma.emotionai',
  appName: 'Razuma',

  // Built by `npm run build:web` (see ../frontend/vite.mobile.config.js).
  // Capacitor copies this directory into the native projects on `cap sync`.
  webDir: '../frontend/dist-mobile',

  server: {
    // Serve the bundle from a local HTTPS origin so camera access
    // (getUserMedia), localStorage and secure-context APIs behave as on the web.
    androidScheme: 'https',
    iosScheme: 'capacitor',
  },

  android: {
    // The API is HTTPS-only; mixed content must stay disabled.
    allowMixedContent: false,
  },
};

export default config;
