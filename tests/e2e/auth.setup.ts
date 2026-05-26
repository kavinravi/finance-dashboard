import { test as setup, expect } from "@playwright/test";
import fs from "node:fs";
import { E2E_PASSWORD, E2E_STATE_PATH } from "./auth.constants";

setup("authenticate", async ({ request }) => {
  fs.mkdirSync("tests/e2e/.auth", { recursive: true });
  const res = await request.post("/api/login", { data: { password: E2E_PASSWORD, next: "/" } });
  expect(res.ok()).toBeTruthy();
  await request.storageState({ path: E2E_STATE_PATH });
});
