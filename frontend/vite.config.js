import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8045",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
