// Explicit acceptance entry; does not redirect the regular 5173/8010 app.
import { defineConfig, mergeConfig } from 'vite'
import base from './vite.config'

export default mergeConfig(base, defineConfig({
  define: { 'import.meta.env.VITE_API_BASE_URL': JSON.stringify('/api') },
  server: { port: 5174, strictPort: true, proxy: { '/api': 'http://127.0.0.1:8011' } },
}))
