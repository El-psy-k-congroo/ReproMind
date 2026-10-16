import { defineConfig, mergeConfig } from 'vite'
import base from './vite.config'

export default mergeConfig(base, defineConfig({
  define: { 'import.meta.env.VITE_API_BASE_URL': JSON.stringify('/api') },
  server: { host: '127.0.0.1', port: 5175, strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8012' } },
}))
