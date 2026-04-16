import { defineConfig, transformWithEsbuild } from 'vite';
import react from '@vitejs/plugin-react';

const jsAsJsx = () => ({
  name: 'js-as-jsx',
  async transform(code, id) {
    if (!/src\/.*\.js$/.test(id)) return null;
    return transformWithEsbuild(code, id, {
      loader: 'jsx',
      jsx: 'automatic',
    });
  },
});

export default defineConfig({
  base: './',
  plugins: [jsAsJsx(), react()],
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.[jt]sx?$/,
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: {
        '.js': 'jsx',
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
  },
  build: {
    outDir: 'build',
  },
});
