import { defineConfig } from "vitest/config";
import path from "node:path";
import { fileURLToPath } from "node:url";
export default defineConfig({
  resolve: { alias: { "@": path.dirname(fileURLToPath(import.meta.url)) } },
  test: { environment: "jsdom", include: ["tests/**/*.test.{ts,tsx}"], restoreMocks: true },
});
