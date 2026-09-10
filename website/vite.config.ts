import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: Object.fromEntries(['/api', '/network', '/rpc', '/healthz'].map(path =>
      [path, { target: process.env.NEUROSHARD_API_URL || 'http://127.0.0.1:38659', changeOrigin: true }]).concat([['/work', { target: process.env.NEUROSHARD_SPONSOR_URL || 'http://127.0.0.1:38660', changeOrigin: true }]])),
  },
})
