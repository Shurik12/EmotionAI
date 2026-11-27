import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const entryDir = path.resolve(__dirname, 'mobile-app');

// Mobile (Capacitor) build.
//
// This is a second, independent Vite build of the same source tree: it renders
// the product screens from `src/mobile/` (no marketing landing) into
// `dist-mobile/`, which Capacitor copies into the Android/iOS shells.
//
// Differences from the web build (`vite.config.js`):
//   * `root` is `mobile-app/` so Vite emits an `index.html` (Capacitor's local
//     server resolves `/` to `index.html`; an HTML entry named `mobile.html`
//     would not be found).
//   * `base: './'` keeps asset URLs relative, since the bundle is loaded from a
//     local WebView origin rather than from `/`.
//   * `envDir` stays at `frontend/` so `.env.mobile` is picked up.
//   * `VITE_API_URL` must be absolute and HTTPS (see `.env.mobile`): inside the
//     app a relative `/api` would target the WebView origin, not the server.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, '');

  return {
    root: entryDir,
    base: './',
    envDir: __dirname,

    plugins: [react()],

    resolve: {
      extensions: ['.js', '.jsx', '.json'],
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@context': path.resolve(__dirname, './src/context'),
        '@styles': path.resolve(__dirname, './src/styles'),
      },
    },

    // Handy for testing the mobile shell in a phone browser on the LAN:
    //   npm run dev:mobile -- --host
    server: {
      port: 3100,
      host: false,
      fs: { allow: [__dirname] },
      proxy: {
        '/api': {
          target: env.API_URL || 'http://localhost:5000',
          changeOrigin: true,
          secure: false,
        },
      },
    },

    build: {
      outDir: path.resolve(__dirname, 'dist-mobile'),
      emptyOutDir: true,
      assetsDir: 'static',
      sourcemap: false,
      minify: 'esbuild',
      // Safari 14.3 (the oldest WKWebView with getUserMedia) is the floor.
      target: 'es2019',
      rollupOptions: {
        input: { index: path.resolve(entryDir, 'index.html') },
      },
      chunkSizeWarningLimit: 512,
    },

    esbuild: {
      loader: 'jsx',
      include: /src\/.*\.jsx?$/,
      jsx: 'automatic',
      drop: mode === 'production' || mode === 'mobile' ? ['console', 'debugger'] : [],
    },

    publicDir: path.resolve(__dirname, 'public'),
  };
});
