import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ['xixi-ms-7e02.tail8331b7.ts.net'],
    proxy: Object.fromEntries(
      ['/api', '/ui', '/media', '/healthz'].map((path) => [path, 'http://127.0.0.1:8000']),
    ),
  },
})
