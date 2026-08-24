import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const isProduction = mode === 'production';

  return {
    root: path.resolve(__dirname),
    base: '/',
    
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

    server: {
      port: 3000,
      open: true,
      proxy: {
        '/api': {
          target: env.API_URL || 'http://localhost:5000',
          changeOrigin: true,
          secure: false,
        },
      },
    },

    build: {
      outDir: 'dist',
      assetsDir: 'static',
      sourcemap: true,
      minify: 'esbuild',
      rollupOptions: {
        input: {
          main: path.resolve(__dirname, 'index.html'),
        },
        output: {
          manualChunks: {
            'react-vendor': ['react', 'react-dom', 'react-router-dom'],
            'vendor': ['axios'],
          },
        },
      },
      chunkSizeWarningLimit: 512,
    },

    optimizeDeps: {
      include: ['react', 'react-dom', 'react-router-dom', 'axios'],
    },

    esbuild: {
      loader: 'jsx',
      include: /src\/.*\.jsx?$/,
      jsx: 'automatic',
      drop: isProduction ? ['console', 'debugger'] : [],
    },

    publicDir: 'public',
  };
});