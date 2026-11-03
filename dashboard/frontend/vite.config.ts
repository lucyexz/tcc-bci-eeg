import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// No "npm run dev" (porta 5173), a API e as figuras vêm do backend em 8000.
export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1000 }, // o recharts sozinho passa de 500 kB
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/figuras": "http://127.0.0.1:8000",
    },
  },
});
