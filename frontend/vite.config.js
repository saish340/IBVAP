import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API and WebSocket calls to the FastAPI backend so
// the frontend can simply fetch("/api/...") and new WebSocket("/ws/...").
const apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
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
