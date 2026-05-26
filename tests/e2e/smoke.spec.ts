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

test("ticker page renders the news section and a memo from an intercepted response", async ({ page }) => {
  await page.route("**/api/memo/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        citedArticles: [{ id: "u1", title: "Source one", url: "https://ex.com/1", source: "finnhub" }],
        memo: {
          ticker: "NVDA", date: "2026-05-25", one_sentence_takeaway: "Fixture takeaway for NVDA.",
          bullish_developments: [{ claim: "Demand strong", why_it_matters: "revenue", source_article_ids: ["u1"], confidence: "medium" }],
          bearish_developments: [], neutral_or_operational_updates: [], watch_items: [], caveats: [],
          overall_news_tone: { label: "somewhat_bullish", score: 64, rationale: "Positive coverage." },
          generatedAt: new Date().toISOString(), model: "gemini-3.5-flash", basedOnArticleCount: 1,
        },
      }),
    }),
  );

  await page.goto("/ticker/NVDA");
  await expect(page.getByText("Fixture takeaway for NVDA.")).toBeVisible();
  await expect(page.getByText(/News Tone/).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent news" })).toBeVisible();
});
