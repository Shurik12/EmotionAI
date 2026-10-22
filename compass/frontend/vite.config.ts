import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// The build goes straight into the landing's dist as a subdirectory, so the
// existing deploy (scp frontend/dist/* to the VM) ships it to
// https://razuma.tech/compass/ without touching nginx or the deploy scripts.
//
// `base: './'` plus hash routing means no SPA rewrite is required. That is also a
// constraint: relative asset URLs resolve against the document URL, so every route
// must stay on `compass/index.html` (i.e. `#/...`). Adding a pushState route such as
// `/compass/screens/P03` would look for its assets under `/compass/screens/`.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: './',
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  build: {
    outDir: fileURLToPath(new URL('../../frontend/dist/compass', import.meta.url)),
    emptyOutDir: true,
    // No .map files in the shipped build: the landing sets `sourcemap: true`, which
    // publishes 428 KB of original source on razuma.tech. Not doing that here.
    sourcemap: false,
    minify: 'esbuild',
    target: 'es2020',
    chunkSizeWarningLimit: 512,
  },
  esbuild: { drop: mode === 'production' ? ['console', 'debugger'] : [] },
}));
