import { test, expect } from "@playwright/test";

test("toggling MA200 removes its line; re-checking restores it", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await expect(page.locator("svg .recharts-line").first()).toBeVisible(); // wait for charts to render
  const ma200 = page.getByRole("checkbox", { name: /MA200/i });
  await expect(ma200).toBeChecked();
  const before = await page.locator("svg .recharts-line").count();
  await ma200.uncheck();
  await expect(page.locator("svg .recharts-line")).toHaveCount(before - 1);
  await ma200.check();
  await expect(page.locator("svg .recharts-line")).toHaveCount(before);
});

test("watchlist is isolated per profile", async ({ page }) => {
  const uniq = `iso${Date.now() % 100000}`;
  // Create + select a throwaway profile via the API (page.request shares browser cookies).
  const created = await (await page.request.post("/api/profiles", { data: { name: uniq } })).json();
  const isoId: string = created.profile.id;
  await page.request.post("/api/profile/select", { data: { id: isoId } });

  // Add a ticker under the iso profile.
  await page.goto("/watchlist");
  await page.getByPlaceholder(/Add ticker/i).fill("QQ");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.locator("a.font-mono", { hasText: /^QQ$/ })).toBeVisible();

  // Switch to the "e2e" profile via the header switcher (opens dropdown, then pick).
  await page.getByRole("button", { name: new RegExp(uniq) }).click();
  await page.getByRole("button", { name: "e2e", exact: true }).click(); // triggers full reload
  await page.waitForURL(/\/watchlist/);

  // The iso profile's ticker must NOT appear under "e2e".
  await expect(page.locator("a.font-mono", { hasText: /^QQ$/ })).toHaveCount(0);

  // Cleanup: deleting the iso profile cascades its watchlist (QQ).
  await page.request.delete(`/api/profiles/${isoId}`);
});

test("deleting the active profile recovers gracefully (no stale-cookie 500)", async ({ page }) => {
  const uniq = `del${Date.now() % 100000}`;
  // create a profile and make it the active one
  const created = await (await page.request.post("/api/profiles", { data: { name: uniq } })).json();
  const id: string = created.profile.id;
  await page.request.post("/api/profile/select", { data: { id } });

  // delete the currently-active profile → cookie is now stale
  await page.request.delete(`/api/profiles/${id}`);

  // a profile-scoped page must NOT 500; it should send us to the picker
  await page.goto("/watchlist");
  await expect(page).toHaveURL(/\/select-profile/);
  await expect(page.getByRole("heading", { name: /who.s looking/i })).toBeVisible();

  // and the watchlist API returns 409 (not 500) for the stale cookie
  const res = await page.request.post("/api/watchlist", { data: { ticker: "QQ" } });
  expect(res.status()).toBe(409);
});
