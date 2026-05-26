import { test, expect } from "@playwright/test";

test("search NVIDIA → NVDA page renders a chart", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder(/Search ticker/i).fill("NVIDIA");
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByRole("button", { name: /NVDA/ }).first().click();
  await expect(page).toHaveURL(/\/ticker\/NVDA/i);
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();
});

test("compare NVDA vs AMD renders an overlay chart", async ({ page }) => {
  await page.goto("/compare?primary=NVDA&comparison=AMD&range=1y");
  await expect(page.getByText(/Relative return/i)).toBeVisible();
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();
});
