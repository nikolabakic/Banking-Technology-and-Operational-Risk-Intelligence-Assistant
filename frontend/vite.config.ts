import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { analyzer } from "vite-bundle-analyzer";

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    mode === "analyze" && analyzer({
      analyzerMode: "static",
      fileName: "bundle-report",
      openAnalyzer: false,
      reportTitle: "BankScope frontend bundle",
    }),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
}));
