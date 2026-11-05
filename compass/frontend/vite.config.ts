import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// The build goes straight into the landing's dist as a subdirectory, so the
// existing deploy (scp frontend/dist/* to the VM) ships it to
// https://razuma.tech/compass/ without touching nginx or the deploy scripts.
//
// `base: '/compass/'` makes all asset URLs absolute (e.g. /compass/assets/...),
// so a hard refresh on a deep history route like /compass/P03 still
// resolves assets correctly — the server's /compass/ SPA fallback then serves
// compass/index.html, which loads /compass/assets/*.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: '/compass/',
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
