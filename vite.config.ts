import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  // Relative base so the built bundle works from any CDN path.
  base: './',
  plugins: [vue()],
  optimizeDeps: {
    // Pre-bundling breaks manifold's import.meta.url based wasm lookup.
    // opencv-js must NOT be excluded: it's CJS and needs the interop shim.
    exclude: ['manifold-3d'],
  },
  build: {
    chunkSizeWarningLimit: 15000,
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
})
