import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// The build goes straight into the landing's dist as a subdirectory, so the
// existing deploy (scp frontend/dist/* to the VM) ships it to
// https://razuma.tech/compass/ without touching nginx or the deploy scripts.
// `base: './'` plus hash routing means no SPA rewrite is required.
export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  build: {
    outDir: fileURLToPath(new URL('../../frontend/dist/compass', import.meta.url)),
    emptyOutDir: true,
  },
});
