import { test as setup, expect } from "@playwright/test";
import fs from "node:fs";
import { E2E_PASSWORD, E2E_STATE_PATH } from "./auth.constants";

setup("authenticate", async ({ request }) => {
  fs.mkdirSync("tests/e2e/.auth", { recursive: true });
  const login = await request.post("/api/login", { data: { password: E2E_PASSWORD, next: "/" } });
  expect(login.ok()).toBeTruthy();

  // The profile gate requires an active profile cookie. Ensure a known "e2e" profile
  // exists (find-or-create) and select it, so the saved state carries fd_profile.
  const listed = await (await request.get("/api/profiles")).json();
  let e2e = (listed.profiles ?? []).find((p: { id: string; name: string }) => p.name === "e2e");
  if (!e2e) {
    const created = await request.post("/api/profiles", { data: { name: "e2e" } });
    expect(created.ok()).toBeTruthy();
    e2e = (await created.json()).profile;
  }
  const sel = await request.post("/api/profile/select", { data: { id: e2e.id } });
  expect(sel.ok()).toBeTruthy();

  await request.storageState({ path: E2E_STATE_PATH });
});
