import { defineConfig, devices } from "@playwright/test";
import { E2E_PASSWORD, E2E_SESSION_SECRET, E2E_STATE_PATH } from "./tests/e2e/auth.constants";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:3000" },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], storageState: E2E_STATE_PATH },
      dependencies: ["setup"],
      testIgnore: /auth\.setup\.ts/,
    },
  ],
  // reuseExistingServer is false so the server always carries the gate env below.
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:3000",
    reuseExistingServer: false,
    timeout: 120_000,
    env: { APP_PASSWORD: E2E_PASSWORD, SESSION_SECRET: E2E_SESSION_SECRET },
  },
});
