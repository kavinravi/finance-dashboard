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

test("charts tab shows fundamentals; news tab shows the memo (intercepted)", async ({ page }) => {
  await page.route("**/api/memo/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        citedArticles: [{ id: "u1", title: "Source one", url: "https://ex.com/1", source: "finnhub" }],
        memo: {
          ticker: "NVDA", date: "2026-05-26", one_sentence_takeaway: "Fixture takeaway for NVDA.",
          bullish_developments: [{ claim: "Demand strong", why_it_matters: "revenue", source_article_ids: ["u1"], confidence: "medium" }],
          bearish_developments: [], neutral_or_operational_updates: [], watch_items: [], caveats: [],
          overall_news_tone: { label: "somewhat_bullish", score: 64, rationale: "Positive coverage." },
          generatedAt: new Date().toISOString(), model: "gemini-3.5-flash", basedOnArticleCount: 1,
        },
      }),
    }),
  );
  await page.route("**/api/fundamentals/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok", source: "sec_edgar",
        asOf: { fiscalYear: 2024, incomePeriodEnd: "2024-09-28", balanceSheetAsOf: "2024-12-28", filingForm: "10-K", filedAt: "2024-11-01", edgarUrl: "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K" },
        view: { marketCap: 3420000000000, peRatio: 28.41, psRatio: 8.7, grossMargin: 0.462, roe: 1.5, roa: 0.28, operatingIncome: 123216000000, currentRatio: 0.92, debtToEquity: 4.15, assets: 364980000000, liabilities: 308030000000, equity: 56950000000, revenue: 391035000000, netIncome: 93736000000, eps: 6.08 },
      }),
    }),
  );

  await page.goto("/ticker/NVDA");
  await expect(page.getByRole("heading", { name: "Fundamentals" })).toBeVisible();
  await expect(page.getByText("$3.42T")).toBeVisible();
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();

  await page.getByRole("link", { name: /News & Memo/i }).click();
  await expect(page).toHaveURL(/\/ticker\/NVDA\/news/);
  await expect(page.getByText("Fixture takeaway for NVDA.")).toBeVisible();
  await expect(page.getByText(/News Tone/).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent news" })).toBeVisible();
});

test("watchlist add then remove; ticker links to its page", async ({ page }) => {
  await page.goto("/watchlist");
  await page.getByPlaceholder(/Add ticker/i).fill("ZZ");
  await page.getByRole("button", { name: "Add" }).click();
  const chip = page.locator("a.font-mono", { hasText: /^ZZ$/ });
  await expect(chip).toBeVisible();
  await expect(chip).toHaveAttribute("href", "/ticker/ZZ");

  await page.getByRole("button", { name: "Remove ZZ" }).click();
  await expect(page.locator("a.font-mono", { hasText: /^ZZ$/ })).toHaveCount(0);
});

test("charts tab has range controls and ALL renders the chart", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  for (const label of ["1M", "3M", "6M", "YTD", "1Y", "ALL"]) {
    await expect(page.getByRole("button", { name: label, exact: true })).toBeVisible();
  }
  await expect(page.locator('input[type="date"]')).toHaveCount(2);
  await page.getByRole("button", { name: "ALL", exact: true }).click();
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();
});

test("ticker tabs are Charts, News, Watchlist in order; Watchlist navigates", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await expect(page.locator("main nav a")).toHaveText([
    "Charts & Fundamentals", "News & Memo", "Watchlist",
  ]);
  await page.locator("main nav").getByRole("link", { name: "Watchlist", exact: true }).click();
  await expect(page).toHaveURL(/\/watchlist/);
  await expect(page.getByRole("heading", { name: "Watchlist" })).toBeVisible();
});

test("header search is hidden on the homepage and present on ticker pages", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("header input")).toHaveCount(0);
  await expect(page.getByPlaceholder(/Search ticker/i)).toBeVisible();

  await page.goto("/ticker/NVDA");
  await expect(page.locator("header input")).toHaveCount(1);
});
