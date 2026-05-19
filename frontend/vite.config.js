import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/pose": {
        target: "ws://localhost:8000",
        ws: true,
      },
      "/health": {
        target: "http://localhost:8000",
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
