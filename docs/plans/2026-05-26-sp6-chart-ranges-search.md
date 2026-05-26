# SP6 — Chart Ranges, Watchlist Tab, Global Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add customizable chart date ranges (1M/3M/6M/YTD/1Y/ALL presets + a custom calendar range over max-available history), move Watchlist into the ticker tab bar between Charts and News, and add a persistent global search header.

**Architecture:** `price-service` fetches max-available history (full on a shallow/cold cache, incremental once the cache is deep) and ships the full bars + indicator series to the Charts tab; a pure `sliceByRange` slices both to the selected window client-side (indicators computed on the full series stay correct at the left edge). A `SiteHeader` (search + nav) replaces the per-page nav, with the search hidden on the homepage.

**Tech Stack:** Next.js 16 (App Router), TypeScript, Recharts, native `<input type="date">`, Vitest, Playwright.

**Spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §11.

**Standing constraints:**
- No AI-authorship traces in commits/docs/branches (no "Claude"/"Anthropic"/"Co-Authored-By"/tool names). Git author `kavinravi` is correct.
- Do NOT run `pnpm lint`/eslint locally (OOM-crashes); verified in CI. Next 16 doesn't lint during `build`.
- Middleware convention is `proxy.ts` (not `middleware.ts`). The E2E suite runs behind the password gate via a saved `storageState` (automatic).
- Tests: `pnpm test` (Vitest; integration hits live Neon via `dotenv/config`), `pnpm test:e2e` (Playwright). Type check: `pnpm exec tsc --noEmit` (ignore errors under `.next/` — stale dev artifacts; a clean check is `rm -rf .next && pnpm build`).
- After the build, do a live smoke + screenshots before declaring done.

**Verified context (2026-05-26):**
- `lib/services/price-service.ts` `getTickerData(ticker, _range="1y")` reads cached bars, refetches when `!isFresh` from a fixed 2-years-ago `from`, upserts, then computes `indicators: { ma10, ma20, ma50, rsi14, macdLine, macdSignal, macdHistogram, volatility5d }` (all arrays parallel to `bars`). The `_range` param is currently ignored. `Range` type is exported and used by `comparison-service` — **keep the param + type** (don't break callers).
- `fmp.dailyPrices(ticker, from, to)` and `yahoo.dailyPrices(ticker, from, to)` both accept ISO `from`/`to` and return ascending `PriceBar[]`. A far-back `from` returns max-available history.
- `app/ticker/[symbol]/page.tsx` (Charts tab) renders header price, `ReturnsTable`, and three inline chart blocks (`PriceChart` ma20/ma50, `RsiChart` rsi14, `MacdChart` macdLine/macdSignal/macdHistogram), the compare form, and `FundamentalsCard`.
- `app/ticker/[symbol]/layout.tsx` has a `← Search` link + symbol `<h1>` + `<TickerTabs/>`.
- `components/ticker-tabs.tsx`: two links — Charts (`/ticker/[symbol]`) and News (`…/news`), active via `usePathname().endsWith("/news")`.
- `components/app-nav.tsx` (rendered by `app/layout.tsx`, alongside `<Analytics/>`): a `"use client"` nav (Home / Watchlist / Health / Log out), hidden on `/login`. `app/layout.tsx` renders `<AppNav/>`.
- `components/search-bar.tsx`: `"use client"`, input + Search button → `GET /api/search?q=` → result buttons that `router.push("/ticker/{symbol}")`. Results `<ul>` is in normal flow (`mt-2`). Placeholder: `Search ticker or company (e.g. NVDA, NVIDIA)`.
- `app/page.tsx` (homepage): centered `<SearchBar/>` + `<RecentSearches/>`.
- `PriceBar` (from `@/lib/types`) has at least `date: string` and `close: number`.

---

## File Structure

**Create:**
- `lib/charts/range.ts` — pure `ChartRange` type, `rangeStartDate`, `sliceByRange`, `SliceableIndicators`.
- `lib/charts/range.test.ts` — unit tests.
- `lib/services/price-fetch-window.ts` — pure `fetchFromDate(bars, today)` + `HISTORY_FLOOR`.
- `lib/services/price-fetch-window.test.ts` — unit tests.
- `components/ticker-charts.tsx` — client range controls + the three charts.
- `components/site-header.tsx` — global search + nav header.

**Modify:**
- `lib/services/price-service.ts` — use `fetchFromDate` for the refetch window.
- `app/ticker/[symbol]/page.tsx` — replace the three inline chart blocks with `<TickerCharts/>`.
- `components/ticker-tabs.tsx` — add the middle Watchlist tab.
- `components/search-bar.tsx` — make the results dropdown an overlay (absolute) so it works in the header.
- `app/layout.tsx` — render `<SiteHeader/>` instead of `<AppNav/>`.
- `app/ticker/[symbol]/layout.tsx` — remove the `← Search` link.
- `tests/e2e/smoke.spec.ts` — add range / tab / header-search coverage.

**Delete:**
- `components/app-nav.tsx` — replaced by `site-header.tsx`.

---

## Task 1: Max-available price history (pure fetch window + wire-in)

**Files:**
- Create: `lib/services/price-fetch-window.ts`
- Test: `lib/services/price-fetch-window.test.ts`
- Modify: `lib/services/price-service.ts`

- [ ] **Step 1: Write the failing unit test** — Create `lib/services/price-fetch-window.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { fetchFromDate, HISTORY_FLOOR } from "./price-fetch-window";
import type { PriceBar } from "@/lib/types";

const bar = (date: string): PriceBar => ({ date, open: 1, high: 1, low: 1, close: 1, adjClose: null, volume: 0 });

describe("fetchFromDate", () => {
  const today = "2026-05-26";

  it("fetches full history when the cache is empty", () => {
    expect(fetchFromDate([], today)).toBe(HISTORY_FLOOR);
  });

  it("fetches full history when the cache is shallow (earliest bar within 3y)", () => {
    expect(fetchFromDate([bar("2024-06-01"), bar("2026-05-20")], today)).toBe(HISTORY_FLOOR);
  });

  it("fetches incrementally (last bar minus a buffer) once the cache is deep", () => {
    // earliest bar older than 3y → deep; last bar 2026-05-20 → from = 5 days earlier
    expect(fetchFromDate([bar("2018-01-02"), bar("2026-05-20")], today)).toBe("2026-05-15");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails** — Run: `pnpm exec vitest run lib/services/price-fetch-window.test.ts` — Expected: FAIL (`Cannot find module './price-fetch-window'`).

- [ ] **Step 3: Implement `lib/services/price-fetch-window.ts`**

```ts
import type { PriceBar } from "@/lib/types";

export const HISTORY_FLOOR = "1970-01-01";
const DEEP_YEARS = 3;
const REVISION_BUFFER_DAYS = 5;

// Decides the `from` date for a refetch.
// - Cold cache, or a cache that doesn't yet reach back DEEP_YEARS → fetch full history (so "ALL" really means all,
//   and pre-existing shallow (e.g. 2-year) caches backfill on their next refresh).
// - Once we hold bars older than DEEP_YEARS → refetch incrementally from a few days before the last bar.
export function fetchFromDate(bars: PriceBar[], today: string): string {
  const deepCutoff = new Date(`${today}T00:00:00Z`);
  deepCutoff.setUTCFullYear(deepCutoff.getUTCFullYear() - DEEP_YEARS);
  const deepCutoffIso = deepCutoff.toISOString().slice(0, 10);

  if (bars.length === 0 || bars[0].date > deepCutoffIso) return HISTORY_FLOOR;

  const last = new Date(`${bars[bars.length - 1].date}T00:00:00Z`);
  last.setUTCDate(last.getUTCDate() - REVISION_BUFFER_DAYS);
  return last.toISOString().slice(0, 10);
}
```

- [ ] **Step 4: Run the test to verify it passes** — Run: `pnpm exec vitest run lib/services/price-fetch-window.test.ts` — Expected: PASS.

- [ ] **Step 5: Wire it into `lib/services/price-service.ts`** — Add the import near the top:

```ts
import { fetchFromDate } from "@/lib/services/price-fetch-window";
```

Replace the stale NOTE comment on the line above `export async function getTickerData` with:

```ts
// History is fetched max-available (see fetchFromDate). `range` is applied client-side (lib/charts/range.ts),
// so it stays accepted-but-unused here to keep callers (e.g. comparison-service) unchanged.
```

Inside `getTickerData`, replace these three lines:

```ts
    const from = new Date(); from.setUTCFullYear(from.getUTCFullYear() - 2);
    const fromIso = from.toISOString().slice(0, 10);
    const toIso = new Date().toISOString().slice(0, 10);
```

with:

```ts
    const toIso = new Date().toISOString().slice(0, 10);
    const fromIso = fetchFromDate(bars, toIso);
```

(Everything else — the FMP/Yahoo fetch, `upsertBars`, the indicator computation — stays unchanged. Keep the `_range` parameter and the `Range` type.)

- [ ] **Step 6: Type-check + confirm existing service tests pass** — Run: `pnpm exec tsc --noEmit` then `pnpm exec vitest run lib/services` — Expected: tsc clean; existing service tests pass.

- [ ] **Step 7: Commit**

```bash
git add lib/services/price-fetch-window.ts lib/services/price-fetch-window.test.ts lib/services/price-service.ts
git commit -m "Fetch max-available price history with incremental refresh"
```

---

## Task 2: Pure range slicing (`lib/charts/range.ts`)

**Files:**
- Create: `lib/charts/range.ts`
- Test: `lib/charts/range.test.ts`

- [ ] **Step 1: Write the failing unit test** — Create `lib/charts/range.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { rangeStartDate, sliceByRange, type SliceableIndicators } from "./range";
import type { PriceBar } from "@/lib/types";

const bar = (date: string, close: number): PriceBar => ({ date, open: close, high: close, low: close, close, adjClose: null, volume: 0 });

// Five consecutive daily bars.
const bars: PriceBar[] = [
  bar("2026-05-18", 10), bar("2026-05-19", 11), bar("2026-05-20", 12), bar("2026-05-21", 13), bar("2026-05-22", 14),
];
const ind: SliceableIndicators = {
  ma20: [1, 2, 3, 4, 5], ma50: [1, 2, 3, 4, 5], rsi14: [1, 2, 3, 4, 5],
  macdLine: [1, 2, 3, 4, 5], macdSignal: [1, 2, 3, 4, 5], macdHistogram: [1, 2, 3, 4, 5],
};

describe("rangeStartDate", () => {
  const wide = [bar("2021-01-04", 1), bar("2026-05-26", 2)];
  it("computes preset cutoffs against today", () => {
    expect(rangeStartDate("1y", wide, "2026-05-26")).toBe("2025-05-26");
    expect(rangeStartDate("6m", wide, "2026-05-26")).toBe("2025-11-26");
    expect(rangeStartDate("ytd", wide, "2026-05-26")).toBe("2026-01-01");
    expect(rangeStartDate("all", wide, "2026-05-26")).toBe("2021-01-04");
  });
  it("clamps a custom start before the first bar to the first bar", () => {
    expect(rangeStartDate({ from: "2019-01-01", to: "2026-05-26" }, wide, "2026-05-26")).toBe("2021-01-04");
  });
});

describe("sliceByRange", () => {
  it("slices bars and every indicator array to a custom window", () => {
    const out = sliceByRange(bars, ind, { from: "2026-05-19", to: "2026-05-21" }, "2026-05-22");
    expect(out.bars.map((b) => b.date)).toEqual(["2026-05-19", "2026-05-20", "2026-05-21"]);
    expect(out.indicators.ma20).toEqual([2, 3, 4]);
    expect(out.indicators.macdHistogram).toEqual([2, 3, 4]);
  });
  it("returns everything for 'all'", () => {
    const out = sliceByRange(bars, ind, "all", "2026-05-22");
    expect(out.bars).toHaveLength(5);
    expect(out.indicators.rsi14).toHaveLength(5);
  });
  it("handles empty input", () => {
    const out = sliceByRange([], ind, "1y", "2026-05-22");
    expect(out.bars).toEqual([]);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails** — Run: `pnpm exec vitest run lib/charts/range.test.ts` — Expected: FAIL (`Cannot find module './range'`).

- [ ] **Step 3: Implement `lib/charts/range.ts`**

```ts
import type { PriceBar } from "@/lib/types";

export type ChartRange = "1m" | "3m" | "6m" | "ytd" | "1y" | "all" | { from: string; to: string };

export type SliceableIndicators = {
  ma20: (number | null)[];
  ma50: (number | null)[];
  rsi14: (number | null)[];
  macdLine: number[];
  macdSignal: number[];
  macdHistogram: number[];
};

function bounds(range: ChartRange, bars: PriceBar[], today: string): { start: string; end: string } {
  const first = bars[0]?.date ?? today;
  const last = bars[bars.length - 1]?.date ?? today;
  if (typeof range === "object") {
    return { start: range.from < first ? first : range.from, end: range.to > last ? last : range.to };
  }
  if (range === "all") return { start: first, end: last };
  if (range === "ytd") return { start: `${today.slice(0, 4)}-01-01`, end: last };
  const months = { "1m": 1, "3m": 3, "6m": 6, "1y": 12 }[range];
  const d = new Date(`${today}T00:00:00Z`);
  d.setUTCMonth(d.getUTCMonth() - months);
  return { start: d.toISOString().slice(0, 10), end: last };
}

export function rangeStartDate(range: ChartRange, bars: PriceBar[], today: string): string {
  return bounds(range, bars, today).start;
}

export function sliceByRange(
  bars: PriceBar[],
  ind: SliceableIndicators,
  range: ChartRange,
  today: string,
): { bars: PriceBar[]; indicators: SliceableIndicators } {
  if (bars.length === 0) return { bars, indicators: ind };
  const { start, end } = bounds(range, bars, today);
  let lo = bars.findIndex((b) => b.date >= start);
  if (lo === -1) lo = bars.length;
  let hiIdx = bars.length - 1;
  while (hiIdx >= 0 && bars[hiIdx].date > end) hiIdx--;
  const hi = hiIdx + 1;
  return {
    bars: bars.slice(lo, hi),
    indicators: {
      ma20: ind.ma20.slice(lo, hi), ma50: ind.ma50.slice(lo, hi), rsi14: ind.rsi14.slice(lo, hi),
      macdLine: ind.macdLine.slice(lo, hi), macdSignal: ind.macdSignal.slice(lo, hi), macdHistogram: ind.macdHistogram.slice(lo, hi),
    },
  };
}
```

- [ ] **Step 4: Run the test to verify it passes** — Run: `pnpm exec vitest run lib/charts/range.test.ts` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/charts/range.ts lib/charts/range.test.ts
git commit -m "Add pure chart range slicing"
```

---

## Task 3: Range controls component + wire into the charts page

**Files:**
- Create: `components/ticker-charts.tsx`
- Modify: `app/ticker/[symbol]/page.tsx`

- [ ] **Step 1: Implement `components/ticker-charts.tsx`**

```tsx
"use client";
import { useState } from "react";
import { PriceChart } from "./price-chart";
import { RsiChart } from "./rsi-chart";
import { MacdChart } from "./macd-chart";
import { sliceByRange, type ChartRange, type SliceableIndicators } from "@/lib/charts/range";
import type { PriceBar } from "@/lib/types";

const PRESETS = [
  { key: "1m", label: "1M" }, { key: "3m", label: "3M" }, { key: "6m", label: "6M" },
  { key: "ytd", label: "YTD" }, { key: "1y", label: "1Y" }, { key: "all", label: "ALL" },
] as const;

export function TickerCharts({ bars, indicators }: { bars: PriceBar[]; indicators: SliceableIndicators }) {
  const [range, setRange] = useState<ChartRange>("1y");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const today = new Date().toISOString().slice(0, 10);
  const sliced = sliceByRange(bars, indicators, range, today);

  const presetActive = (k: string) => typeof range !== "object" && range === k;
  const btn = (active: boolean) =>
    `rounded px-2 py-1 text-xs ${active ? "bg-neutral-200 text-neutral-900" : "bg-neutral-900 text-neutral-300 ring-1 ring-neutral-800 hover:bg-neutral-800"}`;

  function applyCustom(nextFrom: string, nextTo: string) {
    setFrom(nextFrom);
    setTo(nextTo);
    if (nextFrom && nextTo) setRange({ from: nextFrom, to: nextTo });
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => (
          <button key={p.key} onClick={() => setRange(p.key)} className={btn(presetActive(p.key))}>{p.label}</button>
        ))}
        <span className="ml-2 flex items-center gap-1 text-xs text-neutral-500">
          <input type="date" value={from} max={to || today} onChange={(e) => applyCustom(e.target.value, to)}
            className="rounded bg-neutral-900 px-2 py-1 text-neutral-200 ring-1 ring-neutral-800 [color-scheme:dark]" />
          <span>→</span>
          <input type="date" value={to} min={from} max={today} onChange={(e) => applyCustom(from, e.target.value)}
            className="rounded bg-neutral-900 px-2 py-1 text-neutral-200 ring-1 ring-neutral-800 [color-scheme:dark]" />
        </span>
      </div>

      <div><PriceChart bars={sliced.bars} ma20={sliced.indicators.ma20} ma50={sliced.indicators.ma50} /></div>

      <h2 className="mt-6 text-sm font-medium text-neutral-400">RSI (14)</h2>
      <div className="mt-2"><RsiChart bars={sliced.bars} rsi14={sliced.indicators.rsi14} /></div>

      <h2 className="mt-6 text-sm font-medium text-neutral-400">MACD (12/26/9)</h2>
      <div className="mt-2">
        <MacdChart bars={sliced.bars} macdLine={sliced.indicators.macdLine} macdSignal={sliced.indicators.macdSignal} macdHistogram={sliced.indicators.macdHistogram} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Rewrite `app/ticker/[symbol]/page.tsx`** to use it (replaces the three inline chart blocks + their imports):

```tsx
import { getTickerData } from "@/lib/services/price-service";
import { ReturnsTable } from "@/components/returns-table";
import { TickerCharts } from "@/components/ticker-charts";
import { StalenessBadge } from "@/components/staleness-badge";
import { FundamentalsCard } from "@/components/fundamentals-card";
import { formatPrice } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function TickerChartsPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const data = await getTickerData(symbol.toUpperCase(), "1y");
  const latest = data.bars.at(-1)?.close ?? null;

  if (data.bars.length === 0) {
    return (
      <p className="mt-8">
        Couldn&apos;t resolve <span className="font-mono">{symbol.toUpperCase()}</span>. Try another ticker.
      </p>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-baseline justify-between">
        <div className="text-2xl">{formatPrice(latest)}</div>
        <StalenessBadge stale={data.stale} lastBarDate={data.lastBarDate} source={data.source} />
      </div>

      <div className="mt-4"><ReturnsTable returns={data.returns} /></div>

      <div className="mt-6">
        <TickerCharts
          bars={data.bars}
          indicators={{
            ma20: data.indicators.ma20, ma50: data.indicators.ma50, rsi14: data.indicators.rsi14,
            macdLine: data.indicators.macdLine, macdSignal: data.indicators.macdSignal, macdHistogram: data.indicators.macdHistogram,
          }}
        />
      </div>

      <form action="/compare" className="mt-8 flex items-center gap-2">
        <input type="hidden" name="primary" value={data.ticker} />
        <label className="text-sm text-neutral-400">Compare against</label>
        <input name="comparison" placeholder="e.g. SPY"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900">Go</button>
      </form>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Fundamentals</h2>
        <p className="mb-2 text-xs text-neutral-600">From official SEC filings.</p>
        <FundamentalsCard symbol={data.ticker} />
      </section>
    </div>
  );
}
```

- [ ] **Step 3: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add components/ticker-charts.tsx app/ticker/[symbol]/page.tsx
git commit -m "Add chart range controls to the charts tab"
```

---

## Task 4: Watchlist middle tab

**Files:**
- Modify: `components/ticker-tabs.tsx`

- [ ] **Step 1: Add the middle tab** — Replace the `<nav>` block in `components/ticker-tabs.tsx` with:

```tsx
  return (
    <nav className="mt-4 flex gap-6 border-b border-neutral-800">
      <Link href={base} className={cls(!onNews)}>Charts &amp; Fundamentals</Link>
      <Link href="/watchlist" className={cls(false)}>Watchlist</Link>
      <Link href={`${base}/news`} className={cls(onNews)}>News &amp; Memo</Link>
    </nav>
  );
```

(Watchlist links to the global `/watchlist` and is never the "active" tab on a ticker route, so it always uses the inactive style.)

- [ ] **Step 2: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add components/ticker-tabs.tsx
git commit -m "Add Watchlist as the middle ticker tab"
```

---

## Task 5: Global search header

**Files:**
- Modify: `components/search-bar.tsx` (overlay dropdown)
- Create: `components/site-header.tsx`
- Modify: `app/layout.tsx`
- Modify: `app/ticker/[symbol]/layout.tsx`
- Delete: `components/app-nav.tsx`

- [ ] **Step 1: Make the SearchBar results an overlay** — In `components/search-bar.tsx`, change the outer wrapper `<div className="w-full max-w-xl">` to `<div className="relative w-full max-w-xl">`, and change the results `<ul>` opening tag from:

```tsx
        <ul className="mt-2 divide-y divide-neutral-800 rounded bg-neutral-900 ring-1 ring-neutral-800">
```

to:

```tsx
        <ul className="absolute z-20 mt-1 w-full divide-y divide-neutral-800 rounded bg-neutral-900 shadow-lg ring-1 ring-neutral-800">
```

(So the dropdown overlays instead of pushing page content down — needed in the header, and an improvement on the homepage. Nothing else in the file changes.)

- [ ] **Step 2: Create `components/site-header.tsx`** (replaces `app-nav.tsx`)

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SearchBar } from "./search-bar";

export function SiteHeader() {
  const pathname = usePathname();
  if (pathname === "/login") return null;
  const showSearch = pathname !== "/";

  async function logout() {
    await fetch("/api/logout", { method: "POST" });
    window.location.href = "/login";
  }

  return (
    <header className="flex items-center gap-4 border-b border-neutral-800 px-4 py-2 text-sm">
      {showSearch ? <div className="flex-1"><SearchBar /></div> : <div className="flex-1" />}
      <nav className="flex items-center gap-4">
        <Link href="/" className="text-neutral-300 hover:text-white">Home</Link>
        <Link href="/watchlist" className="text-neutral-300 hover:text-white">Watchlist</Link>
        <Link href="/health" className="text-neutral-300 hover:text-white">Health</Link>
        <button onClick={logout} className="text-neutral-300 hover:text-white">Log out</button>
      </nav>
    </header>
  );
}
```

- [ ] **Step 3: Render it in `app/layout.tsx`** — Replace the `AppNav` import with `SiteHeader` and the `<AppNav />` element with `<SiteHeader />`:

```tsx
import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { Providers } from "./providers";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = { title: "Finance Dashboard", description: "Investing research" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        <Providers>
          <SiteHeader />
          {children}
        </Providers>
        <Analytics />
      </body>
    </html>
  );
}
```

- [ ] **Step 4: Remove the `← Search` link from `app/ticker/[symbol]/layout.tsx`** — Replace the whole file with:

```tsx
import { TickerTabs } from "@/components/ticker-tabs";

export default async function TickerLayout({
  children, params,
}: { children: React.ReactNode; params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const ticker = symbol.toUpperCase();
  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <h1 className="font-mono text-3xl font-semibold">{ticker}</h1>
      <TickerTabs symbol={ticker} />
      {children}
    </main>
  );
}
```

- [ ] **Step 5: Delete the old nav**

```bash
git rm components/app-nav.tsx
```

- [ ] **Step 6: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors (no remaining references to `AppNav`).

- [ ] **Step 7: Commit**

```bash
git add components/search-bar.tsx components/site-header.tsx app/layout.tsx app/ticker/[symbol]/layout.tsx
git commit -m "Add persistent global search header"
```

---

## Task 6: E2E coverage + full verification + live smoke

**Files:**
- Modify: `tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Add SP6 E2E tests** — Append to `tests/e2e/smoke.spec.ts`:

```ts
test("charts tab has range controls and ALL renders the chart", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  for (const label of ["1M", "3M", "6M", "YTD", "1Y", "ALL"]) {
    await expect(page.getByRole("button", { name: label, exact: true })).toBeVisible();
  }
  await expect(page.locator('input[type="date"]')).toHaveCount(2);
  await page.getByRole("button", { name: "ALL", exact: true }).click();
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();
});

test("Watchlist sits between Charts and News and navigates", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await page.getByRole("link", { name: "Watchlist", exact: true }).click();
  await expect(page).toHaveURL(/\/watchlist/);
  await expect(page.getByRole("heading", { name: "Watchlist" })).toBeVisible();
});

test("header search is hidden on the homepage and present on ticker pages", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("header input")).toHaveCount(0);   // homepage: nav only
  await expect(page.getByPlaceholder(/Search ticker/i)).toBeVisible(); // homepage body search still there

  await page.goto("/ticker/NVDA");
  await expect(page.locator("header input")).toHaveCount(1);   // header search present off the homepage
});
```

- [ ] **Step 2: Run the full E2E suite** — Run: `pnpm test:e2e` — Expected: all specs pass (the three new tests + the existing smokes/auth, which are unaffected: the homepage search is unchanged, the News tab link still resolves uniquely, and the watchlist add/remove input placeholder `Add ticker` doesn't collide with the header `Search ticker`). Ensure nothing else is serving port 3000.

- [ ] **Step 3: Full unit + integration suite** — Run: `pnpm test` — Expected: all pass (new `price-fetch-window` + `range` suites included; no regressions).

- [ ] **Step 4: Clean type-check + production build** — Run: `rm -rf .next && pnpm build` — Expected: build succeeds; no deprecation warnings; routes unchanged.

- [ ] **Step 5: Live smoke + screenshots** — Start the gated prod server (`APP_PASSWORD=localtest SESSION_SECRET=$(openssl rand -hex 32) pnpm start`), authenticate in a throwaway Playwright script, and capture: AAPL charts tab at **1Y**, at **ALL** (confirm it shows many years), and with a **custom** from–to; the **header search** visible on the ticker page and absent on `/`; the **three-tab** bar (Charts / Watchlist / News). Confirm AAPL's ALL view extends well beyond 2 years (max-history fetch worked). Stop the server afterward (exit 143 from SIGTERM is expected).

- [ ] **Step 6: Commit (the E2E additions, plus any verification fix)**

```bash
git add tests/e2e/smoke.spec.ts
git commit -m "Add E2E coverage for chart ranges, watchlist tab, and header search"
```

---

## Self-Review (completed during planning)

**1. Spec coverage (§11):**
- §11.2 max history + incremental → Task 1 (`fetchFromDate` with the shallow-cache backfill so existing 2y caches reach full depth — a refinement of the spec's "warm→incremental" that also handles migration). ✅
- §11.3 range slicing + controls → Task 2 (`range.ts`) + Task 3 (`TickerCharts` + page). ✅
- §11.4 watchlist tab → Task 4. ✅
- §11.5 global search header (`SiteHeader`, remove back link, hide on `/`, delete `app-nav`) → Task 5. ✅
- §11.6 error handling (clamping, empty, pure/total slicing) → Task 2 (`bounds`/`sliceByRange` clamp + empty cases, tested). ✅
- §11.7 testing → Tasks 1/2 (unit) + Task 6 (E2E + live smoke). ✅
- §11.8 acceptance → exercised by Task 6.

**2. Placeholder scan:** none — every code step is complete; commands have expected output.

**3. Type consistency:** `SliceableIndicators` (Task 2) is produced by the page (Task 3) from `data.indicators` and consumed by `TickerCharts`/`sliceByRange` identically. `ChartRange` is shared (Task 2 → Task 3). `fetchFromDate(bars, today)` (Task 1) matches its single call site in `price-service`. `SiteHeader` (Task 5) replaces `AppNav` in `layout.tsx`; `app-nav.tsx` is deleted and has no remaining importers. `TickerTabs` keeps its `{ symbol }` prop (Task 4).

---

## Execution Handoff

Execute task-by-task. Recommended: subagent-driven development (fresh subagent per task + review between tasks), with the controller running the full `pnpm test` + `pnpm test:e2e` after Task 5 (the layout/nav change touches global rendering) and personally doing the Task 6 live smoke + screenshots (confirming "ALL" really shows max history is something only a live check catches). Tasks 1 and 2 are TDD with real failing-first tests; Tasks 3–5 are write-then-typecheck (behavior verified by the Task 6 E2E + live smoke).
