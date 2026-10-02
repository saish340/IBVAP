import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev only: the dev server proxies /api + /ws to the FastAPI backend so
// the app can use relative paths. In production the built files are served
// by nginx (same-origin proxy) or by Vercel (VITE_API_URL points at backend).
const apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  // Split deploy (Vercel): build output goes to dist/.
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1200,
  },
  preview: {
    port: 4173,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: apiTarget.replace(/^http/, "ws"),
        ws: true,
      },
    },
  },
});

