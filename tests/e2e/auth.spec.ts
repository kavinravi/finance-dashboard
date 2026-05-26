import { test, expect } from "@playwright/test";
import { E2E_PASSWORD } from "./auth.constants";

test.use({ storageState: { cookies: [], origins: [] } });

test("unauthenticated request is redirected to /login", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await expect(page).toHaveURL(/\/login/);
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("wrong password shows an error and stays on /login", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/password/i).fill("definitely-wrong");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText(/incorrect password/i)).toBeVisible();
  await expect(page).toHaveURL(/\/login/);
});

test("correct password signs in and lands on the homepage", async ({ page }) => {
  await page.goto("/login?next=%2F");
  await page.getByLabel(/password/i).fill(E2E_PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL("http://localhost:3000/");
  await expect(page.getByPlaceholder(/Search ticker/i)).toBeVisible();
});
