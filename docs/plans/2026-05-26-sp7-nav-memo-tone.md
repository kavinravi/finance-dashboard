# SP7 — Navigation, Memo Polish, Tone Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the watchlist navigation dead-end (reorder it last + make watchlist tickers clickable), color the memo's bullish/bearish section headers green/red, and recalibrate + clarify the News Tone score (kept as coverage-tone, never an investment-outlook/advice score).

**Architecture:** Small UI + prompt changes only — no new files, providers, or tables. Reorder/link in the existing tab + watchlist components; add a `tone` prop to the memo's `Group`; tighten the Gemini prompt's score calibration and add a clarifying caption to the tone meter.

**Tech Stack:** Next.js 16 (App Router), TypeScript, Tailwind, Recharts (unaffected), Gemini prompt (`lib/providers/gemini.ts`), Vitest, Playwright.

**Spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §12.

**Standing constraints:**
- No AI-authorship traces in commits/docs/branches (no "Claude"/"Anthropic"/"Co-Authored-By"/tool names). Git author `kavinravi` is correct.
- `gemini-3.5-flash` is the verified 2026 stable model — don't downgrade it.
- **News Tone stays coverage-sentiment, NOT an investment outlook/forecast/advice** — never add buy/sell/hold or outlook framing. The recalibration tightens honesty/range; it does NOT instruct pessimism.
- Do NOT run `pnpm lint`/eslint locally (OOM); Next 16 doesn't lint during build. Tests: `pnpm test` (Vitest; integration hits live Neon), `pnpm test:e2e` (Playwright, gated via saved storageState). Type check: `pnpm exec tsc --noEmit` (ignore stale `.next/` errors).
- After the build, do a **live memo smoke** (real Gemini) before declaring done — the mocked suite can't catch tone calibration.

**Verified context (current code):**
- `components/ticker-tabs.tsx`: nav with `cls(active)` helper; currently order Charts (`base`) · Watchlist (`/watchlist`, `cls(false)`) · News (`${base}/news`). `base = \`/ticker/${symbol}\``, `onNews = pathname.endsWith("/news")`.
- `components/watchlist-manager.tsx`: chips render `<span className="font-mono">{t}</span>` + a remove `<button aria-label={\`Remove ${t}\`}>×</button>`. Does NOT import `Link`.
- `components/memo-card.tsx`: `Group({ title, items, cited })` renders an `<h3 className="text-xs font-medium uppercase text-neutral-500">`; called 3× (Bullish / Bearish / Neutral / operational).
- `lib/providers/gemini.ts` `buildPrompt`: an array of rule lines incl. the `overall_news_tone.score MUST be an INTEGER from 0 to 100 on this scale: 0-30 bearish, 31-45 somewhat bearish, 46-55 neutral, 56-70 somewhat bullish, 71-100 bullish...` line. `generateMemo` is mocked in service tests.
- `components/tone-meter.tsx`: shows "News Tone" + `score/100 · label` + a colored bar; no caption.
- `tests/e2e/smoke.spec.ts`: has `watchlist add then remove updates the chips` (asserts `span.font-mono` `ZZ`) and `Watchlist sits between Charts and News and navigates` (clicks `main nav` Watchlist link). Both need updating for the reorder + link change.

---

## File Structure

**Modify:**
- `components/ticker-tabs.tsx` — reorder Watchlist last.
- `components/watchlist-manager.tsx` — ticker name becomes a `Link` to `/ticker/{t}`.
- `components/memo-card.tsx` — `Group` gains a `tone` prop coloring the header.
- `lib/providers/gemini.ts` — add a score-calibration rule to `buildPrompt`.
- `components/tone-meter.tsx` — add a clarifying caption.
- `tests/e2e/smoke.spec.ts` — update the watchlist + tab tests.

No new files, tests files, deletions, or migrations.

---

## Task 1: Watchlist navigation (reorder last + clickable tickers)

**Files:**
- Modify: `components/ticker-tabs.tsx`
- Modify: `components/watchlist-manager.tsx`
- Modify: `tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Reorder the tabs** — In `components/ticker-tabs.tsx`, replace the `return (...)` nav block with (Watchlist now last):

```tsx
  return (
    <nav className="mt-4 flex gap-6 border-b border-neutral-800">
      <Link href={base} className={cls(!onNews)}>Charts &amp; Fundamentals</Link>
      <Link href={`${base}/news`} className={cls(onNews)}>News &amp; Memo</Link>
      <Link href="/watchlist" className={cls(false)}>Watchlist</Link>
    </nav>
  );
```

- [ ] **Step 2: Make watchlist tickers clickable** — In `components/watchlist-manager.tsx`, add the import at the top (after the existing imports):

```tsx
import Link from "next/link";
```

Then change the chip's ticker span from:

```tsx
            <span className="font-mono">{t}</span>
```

to:

```tsx
            <Link href={`/ticker/${t}`} className="font-mono hover:underline">{t}</Link>
```

(Leave the remove `×` button unchanged.)

- [ ] **Step 3: Update the E2E tests** — In `tests/e2e/smoke.spec.ts`, replace the `watchlist add then remove updates the chips` test with:

```ts
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
```

And replace the `Watchlist sits between Charts and News and navigates` test with:

```ts
test("ticker tabs are Charts, News, Watchlist in order; Watchlist navigates", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await expect(page.locator("main nav a")).toHaveText([
    "Charts & Fundamentals", "News & Memo", "Watchlist",
  ]);
  await page.locator("main nav").getByRole("link", { name: "Watchlist", exact: true }).click();
  await expect(page).toHaveURL(/\/watchlist/);
  await expect(page.getByRole("heading", { name: "Watchlist" })).toBeVisible();
});
```

- [ ] **Step 4: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add components/ticker-tabs.tsx components/watchlist-manager.tsx tests/e2e/smoke.spec.ts
git commit -m "Reorder Watchlist tab last and link watchlist tickers to their pages"
```

(The full E2E run is in Task 4.)

---

## Task 2: Color the memo's bullish/bearish sections

**Files:**
- Modify: `components/memo-card.tsx`

- [ ] **Step 1: Add a `tone` prop to `Group`** — In `components/memo-card.tsx`, replace the `Group` function signature + its `<h3>` with:

```tsx
function Group({ title, items, cited, tone }: { title: string; items: Development[]; cited: CitedArticle[]; tone: "bullish" | "bearish" | "neutral" }) {
  if (items.length === 0) return null;
  const titleColor = tone === "bullish" ? "text-emerald-400" : tone === "bearish" ? "text-red-400" : "text-neutral-500";
  return (
    <div className="mt-4">
      <h3 className={`text-xs font-medium uppercase ${titleColor}`}>{title}</h3>
```

(Leave the rest of `Group` — the `<ul>`/`<li>` body — unchanged.)

- [ ] **Step 2: Pass the tone at the three call sites** — Replace the three `<Group .../>` lines in the rendered memo with:

```tsx
      <Group title="Bullish" items={m.bullish_developments} cited={data.citedArticles} tone="bullish" />
      <Group title="Bearish" items={m.bearish_developments} cited={data.citedArticles} tone="bearish" />
      <Group title="Neutral / operational" items={m.neutral_or_operational_updates} cited={data.citedArticles} tone="neutral" />
```

- [ ] **Step 3: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add components/memo-card.tsx
git commit -m "Color memo bullish/bearish section headers green/red"
```

---

## Task 3: Recalibrate + clarify the News Tone

**Files:**
- Modify: `lib/providers/gemini.ts`
- Modify: `components/tone-meter.tsx`

- [ ] **Step 1: Add a calibration rule to the prompt** — In `lib/providers/gemini.ts` `buildPrompt`, find the existing line that ends with `...Do NOT use a 0-1 scale.\`,` and insert this new line immediately AFTER it:

```ts
    `- Calibrate the score from the EVIDENCE and use the FULL range; do NOT default to ~50/neutral. If coverage emphasizes risks, declines, earnings misses, downgrades, litigation, layoffs, guidance cuts, or controversy, score in the bearish range (0-45). Reserve 56-100 for coverage that is genuinely, predominantly favorable. Most coverage is mixed — score it honestly, not optimistically. This is still COVERAGE tone, never a forecast or recommendation.`,
```

- [ ] **Step 2: Add the clarifying caption to the tone meter** — In `components/tone-meter.tsx`, add a caption line immediately after the closing `</div>` of the bar (the `<div className="mt-1 h-2 w-full rounded bg-neutral-800">…</div>`), before the component's outer closing `</div>`:

```tsx
      <p className="mt-1 text-xs text-neutral-600">Tone of recent news coverage — not the stock&apos;s performance or a forecast.</p>
```

- [ ] **Step 3: Type-check + confirm provider tests still pass** — Run: `pnpm exec tsc --noEmit` then `pnpm exec vitest run lib/providers` — Expected: tsc clean; provider tests pass (the prompt is a string the service mocks; adding a rule line shouldn't break any assertion — if a `buildPrompt` test asserts exact content, update it to include the new line).

- [ ] **Step 4: Commit**

```bash
git add lib/providers/gemini.ts components/tone-meter.tsx
git commit -m "Recalibrate News Tone scoring and clarify it is coverage tone"
```

---

## Task 4: Full verification + live memo smoke

**Files:** none (verification only)

- [ ] **Step 1: Full unit + integration suite** — Run: `pnpm test` — Expected: all pass, no regressions.

- [ ] **Step 2: Full E2E suite** — Run: `pnpm test:e2e` — Expected: all pass, including the updated watchlist + tab-order tests. (Ensure nothing else is serving port 3000.)

- [ ] **Step 3: Clean type-check + production build** — Run: `rm -rf .next && pnpm build` — Expected: build succeeds; no deprecation warnings.

- [ ] **Step 4: Live memo smoke + screenshots** — Start the gated prod server (`APP_PASSWORD=localtest SESSION_SECRET=$(openssl rand -hex 32) pnpm start`), authenticate in a throwaway Playwright script, then:
  - Open `/ticker/META/news`, force-regenerate the memo (the Regenerate button or `/api/memo/META?force=1`) so the recalibrated prompt runs against **live** Gemini; capture the memo. Confirm the News Tone is **evidence-calibrated** (not anchored at ~50 when META's recent coverage is clearly negative) and that the **Bullish header is green / Bearish header is red**, with the **caption** under the tone meter.
  - Repeat for one more ticker (e.g. NVDA) to confirm the calibration isn't simply pessimistic — favorable coverage should still score high.
  - Open `/watchlist`, confirm a ticker chip is a link and clicking it opens `/ticker/{symbol}` (tabs present); confirm the tab order Charts · News · Watchlist.
  Stop the server afterward (exit 143 from SIGTERM is expected).

- [ ] **Step 5: Commit (only if a verification fix was needed)**

```bash
git add -A
git commit -m "SP7 verification fixes"
```

(If no fixes were needed, skip — don't create an empty commit.)

---

## Self-Review (completed during planning)

**1. Spec coverage (§12):**
- §12.2 watchlist nav (reorder last + clickable tickers) → Task 1. ✅
- §12.3 memo section colors → Task 2. ✅
- §12.4 News Tone recalibration + caption → Task 3. ✅
- §12.5 error handling — no new failure modes; the recalibrated prompt validates against the unchanged Zod schema (covered by existing behavior). ✅
- §12.6 testing (E2E watchlist link + tab order; live memo + screenshots) → Task 1 (E2E edits) + Task 4 (run + live). ✅
- §12.7 acceptance → exercised by Task 4.

**2. Placeholder scan:** none — every code step shows the exact change; commands have expected output.

**3. Type consistency:** `Group`'s new `tone: "bullish" | "bearish" | "neutral"` (Task 2) is passed at exactly the three call sites with matching literals. `ticker-tabs.tsx` keeps `base`/`cls`/`onNews` (Task 1). `watchlist-manager.tsx` adds the `Link` import used by the chip (Task 1). The E2E selector change (`span.font-mono` → `a.font-mono`) matches the `<Link>` (renders an `<a>`) from Task 1. `buildPrompt` stays a `string[]`-joined function (Task 3) — only a line is added.

---

## Execution Handoff

Execute task-by-task. Recommended: subagent-driven development (fresh subagent per task + review), with the controller running Task 4 (the live memo smoke — only a real Gemini call confirms the tone calibration, and a screenshot confirms the colors/caption). All four tasks are small; Tasks 1–3 are write-then-typecheck (behavior verified by the Task 4 E2E + live smoke); there's no new pure logic warranting a unit test.
