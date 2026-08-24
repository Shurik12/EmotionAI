import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import checker from 'vite-plugin-checker';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const isDevelopment = mode === 'development';
  const isAnalyze = mode === 'analyze';
  const isProduction = mode === 'production';

  return {
    root: path.resolve(__dirname),
    base: '/',
    
    plugins: [
      // @vitejs/plugin-react handles JSX transformation with esbuild
      // No Babel configuration needed!
      react({
        // This enables fast refresh and JSX transformation
        // All done via esbuild, not Babel
      }),
      isDevelopment && checker({
        typescript: false,
        eslint: {
          lintCommand: 'eslint "./src/**/*.{js,jsx}"',
          dev: { logLevel: ['error'] },
        },
      }),
      isAnalyze && visualizer({
        open: true,
        filename: 'dist/stats.html',
        gzipSize: true,
        brotliSize: true,
      }),
    ].filter(Boolean),

    resolve: {
      extensions: ['.js', '.jsx', '.json'],
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@services': path.resolve(__dirname, './src/services'),
        '@context': path.resolve(__dirname, './src/context'),
        '@styles': path.resolve(__dirname, './src/styles'),
        '@features': path.resolve(__dirname, './src/features'),
      },
    },

    css: {
      modules: {
        localsConvention: 'camelCase',
        generateScopedName: isDevelopment
          ? '[path][name]__[local]--[hash:base64:5]'
          : '[hash:base64:8]',
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
      minify: 'esbuild', // Uses esbuild, not Babel
      rollupOptions: {
        input: {
          main: path.resolve(__dirname, 'index.html'),
        },
        output: {
          manualChunks: (id) => {
            if (id.includes('node_modules')) {
              if (id.includes('react') || id.includes('react-dom') || id.includes('react-router')) {
                return 'react-vendor';
              }
              if (id.includes('axios')) {
                return 'axios';
              }
              if (id.includes('lodash')) {
                return 'lodash';
              }
              return 'vendor';
            }
          },
        },
      },
      chunkSizeWarningLimit: 512,
      cssMinify: true,
      commonjsOptions: {
        transformMixedEsModules: true,
      },
    },

    optimizeDeps: {
      include: ['react', 'react-dom', 'react-router-dom', 'axios'],
    },

    esbuild: {
      loader: 'jsx',
      include: /src\/.*\.jsx?$/,
      exclude: [],
      // Drop console and debugger in production
      drop: isProduction ? ['console', 'debugger'] : [],
      // JSX runtime for React 17+
      jsx: 'automatic',
    },

    publicDir: 'public',
  };
});