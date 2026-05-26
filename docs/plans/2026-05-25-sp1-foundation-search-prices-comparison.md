# SP1 — Foundation + Search + Prices + Comparison Implementation Plan

> **For implementers:** Build this plan task-by-task using test-driven development — write the failing test, run it to confirm it fails, implement the minimal code, run it to confirm it passes, then commit. Steps use checkbox (`- [ ]`) syntax for tracking; review between tasks.

**Goal:** Build the first usable slice of the rebuilt finance dashboard — search a ticker/company name, open a research page with a price chart + period returns + technical indicators, and overlay a one-target comparison — as a single Next.js app backed by a Neon Postgres cache.

**Architecture:** One Next.js 15 App Router app at the repo root. All provider calls happen server-side. FMP (free) is the primary source for search/profile/EOD prices with `yahoo-finance2` as a best-effort fallback; results are cached in Neon Postgres (via Drizzle) as a bounded cache with freshness rules. Technical indicators are pure TypeScript functions computed on the fly from cached bars. Behavior degrades gracefully (FMP → Yahoo → stale cache) and never crashes a page.

**Tech Stack:** Next.js 15 (App Router, TypeScript), Tailwind + shadcn/ui, Drizzle ORM + Neon Postgres, Zod, TanStack Query, Recharts + lightweight-charts, Vitest (unit/integration), Playwright (E2E smoke), pnpm.

Full design context: [`docs/specs/2026-05-25-finance-dashboard-rebuild-design.md`](../specs/2026-05-25-finance-dashboard-rebuild-design.md).

---

## Verified external API shapes (use these exact forms)

- **FMP `/stable/`** (key as `apikey=` query param):
  - Search: `GET https://financialmodelingprep.com/stable/search-symbol?query=AAPL&apikey=KEY` and `GET .../stable/search-name?query=apple&apikey=KEY` → array of `{ symbol, name, currency, exchangeFullName, exchange }`.
  - Profile: `GET .../stable/profile?symbol=AAPL&apikey=KEY` → array of one `{ symbol, companyName, currency, exchange, exchangeFullName, sector, industry, isEtf, isFund, isActivelyTrading, cik, ... }`.
  - Daily EOD: `GET .../stable/historical-price-eod/full?symbol=AAPL&from=2024-01-01&to=2025-01-01&apikey=KEY` → array of bars. **Field casing is unconfirmed — Task 10 includes a live verification step before the parser is finalized.** This endpoint has no `adjClose` (we use `close`).
- **`yahoo-finance2` v3:** `import YahooFinance from "yahoo-finance2"; const yf = new YahooFinance();`
  - Bars: `await yf.chart(symbol, { period1, period2, interval: "1d" })` → `{ quotes: { date: Date, open, high, low, close, volume, adjclose? }[] }` (note `adjclose` is **lowercase**).
  - Search: `await yf.search(query)` → `{ quotes: { symbol, exchange, shortname?, longname?, typeDisp? }[], news: [...] }`.
- **Drizzle + Neon (HTTP):** `import { neon } from "@neondatabase/serverless"; import { drizzle } from "drizzle-orm/neon-http";`
- **Next 15 dynamic route params are a Promise:** `{ params }: { params: Promise<{ symbol: string }> }`, then `const { symbol } = await params;`.

---

## File structure (created across the plan)

```
finance-dashboard/
  app/
    layout.tsx, page.tsx            # homepage (search-first)
    ticker/[symbol]/page.tsx        # research page
    compare/page.tsx
    api/search/route.ts
    api/prices/[symbol]/route.ts
    api/compare/route.ts
    providers.tsx                   # TanStack Query provider
  components/
    search-bar.tsx, recent-searches.tsx
    price-chart.tsx, returns-table.tsx
    comparison-chart.tsx, staleness-badge.tsx
  lib/
    types.ts                        # PriceBar, SearchResult, CompanyProfile, PeriodReturns
    env.ts                          # Zod-validated env
    formatters.ts                   # number/percent/date formatting
    indicators/
      returns.ts, moving-averages.ts, rsi.ts, macd.ts, volatility.ts, index.ts
    providers/
      base.ts, fmp.ts, yahoo.ts
    db/
      client.ts, schema.ts, provider-state.ts, companies.ts, price-bars.ts, recent-searches.ts
    services/
      search-service.ts, price-service.ts, comparison-service.ts
  drizzle/                          # generated migrations
  drizzle.config.ts
  vitest.config.ts
  tests/e2e/smoke.spec.ts
  .env.example
```

---

## Task 0: Prerequisites (manual, one-time)

**Files:** none (environment setup)

- [ ] **Step 1: Confirm tooling**

Run: `node -v && pnpm -v`
Expected: Node ≥ 20 and pnpm ≥ 9. If pnpm is missing: `corepack enable && corepack prepare pnpm@latest --activate`.

- [ ] **Step 2: Create a Neon project and get the connection string**

Go to https://neon.tech → create a project (region close to you) → copy the **pooled** connection string (looks like `postgresql://USER:PASS@ep-xxx-pooler.REGION.aws.neon.tech/neondb?sslmode=require`).

- [ ] **Step 3: Confirm the `.env` already has provider keys**

The repo root `.env` (gitignored) should already contain `FMP_API_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`. You will add `DATABASE_URL` in Task 2. Do not commit `.env`.

---

## Task 1: Scaffold the Next.js app at the repo root

**Files:**
- Create: `package.json`, `tsconfig.json`, `next.config.ts`, `app/layout.tsx`, `app/page.tsx`, Tailwind config, etc. (generated)

The repo root is non-empty (`legacy/`, `docs/`, `plan.md`, `.git`), so `create-next-app .` would abort. Scaffold in a temp dir and copy in.

- [ ] **Step 1: Generate the app in a sibling temp directory**

Run from the repo root:
```bash
cd ..
pnpm create next-app@latest fd-scaffold --ts --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-pnpm
```
Expected: a `fd-scaffold/` folder with a working Next app.

- [ ] **Step 2: Copy generated files into the repo (preserve our .git and .gitignore)**

```bash
rsync -a --exclude '.git' --exclude 'node_modules' --exclude '.gitignore' --exclude '.next' fd-scaffold/ finance-dashboard/
rm -rf fd-scaffold
cd finance-dashboard
```
Expected: `app/`, `package.json`, `next.config.ts`, `tsconfig.json`, `tailwind`/`postcss` configs now exist at the repo root; our `.gitignore` is untouched.

- [ ] **Step 3: Ensure Node/Next ignores are present in `.gitignore`**

Append any missing entries (idempotent):
```bash
for e in 'node_modules' '.next' 'out' '.vercel' 'next-env.d.ts' '.env*.local'; do grep -qxF "$e" .gitignore || echo "$e" >> .gitignore; done
```
Expected: `node_modules`, `.next`, etc. present in `.gitignore` (it already ignores `.env`).

- [ ] **Step 4: Install and run the dev server**

Run: `pnpm install && pnpm dev`
Expected: dev server boots on http://localhost:3000 with the default Next page. Stop it with Ctrl-C.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Scaffold Next.js app at repo root"
```

---

## Task 2: Add dependencies, env validation, and `.env.example`

**Files:**
- Create: `lib/env.ts`, `.env.example`
- Modify: `package.json` (deps)

- [ ] **Step 1: Install runtime + dev dependencies**

```bash
pnpm add drizzle-orm @neondatabase/serverless zod @tanstack/react-query recharts lightweight-charts yahoo-finance2
pnpm add -D drizzle-kit vitest vite-tsconfig-paths dotenv @playwright/test
```
Expected: installs succeed; these appear in `package.json`.

- [ ] **Step 2: Write the env validation module**

Create `lib/env.ts`:
```ts
import { z } from "zod";

const schema = z.object({
  DATABASE_URL: z.string().url(),
  FMP_API_KEY: z.string().min(1),
  FMP_DAILY_LIMIT: z.coerce.number().int().positive().default(250),
  FINNHUB_API_KEY: z.string().min(1).optional(), // used in SP2
  GEMINI_API_KEY: z.string().min(1).optional(),  // used in SP2
});

export const env = schema.parse({
  DATABASE_URL: process.env.DATABASE_URL,
  FMP_API_KEY: process.env.FMP_API_KEY,
  FMP_DAILY_LIMIT: process.env.FMP_DAILY_LIMIT,
  FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
  GEMINI_API_KEY: process.env.GEMINI_API_KEY,
});
```

- [ ] **Step 3: Write `.env.example` (committed) and add `DATABASE_URL` to `.env`**

Create `.env.example`:
```bash
# Database (Neon Postgres) — pooled connection string
DATABASE_URL=

# Market data
FMP_API_KEY=
FMP_DAILY_LIMIT=250

# News + LLM (used starting SP2)
FINNHUB_API_KEY=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
GEMINI_PREVIEW_MODEL=gemini-3-flash-preview

# SEC (used in SP3) — required contact UA
SEC_USER_AGENT="Kavin Ravi kavinravi121@gmail.com"
```
Then add your real `DATABASE_URL` (from Task 0) to the gitignored `.env`.

- [ ] **Step 4: Commit**

```bash
git add package.json pnpm-lock.yaml lib/env.ts .env.example
git commit -m "Add dependencies, env validation, and .env.example"
```

---

## Task 3: Database — schema, client, and migration

**Files:**
- Create: `lib/db/schema.ts`, `lib/db/client.ts`, `drizzle.config.ts`

We use `doublePrecision` for OHLC (avoids Drizzle's `numeric`→string mapping) and `bigint(mode:"number")` for volume (daily volumes are well under 2^53).

- [ ] **Step 1: Write the schema (4 tables)**

Create `lib/db/schema.ts`:
```ts
import {
  pgTable, uuid, text, date, timestamp, doublePrecision, bigint, integer, unique,
} from "drizzle-orm/pg-core";

export const companies = pgTable("companies", {
  id: uuid("id").primaryKey().defaultRandom(),
  ticker: text("ticker").notNull().unique(),
  name: text("name").notNull(),
  assetType: text("asset_type").notNull().default("stock"), // stock | etf | index
  exchange: text("exchange"),
  sector: text("sector"),
  industry: text("industry"),
  currency: text("currency"),
  lastProfileRefreshAt: timestamp("last_profile_refresh_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

export const priceBarsDaily = pgTable("price_bars_daily", {
  id: uuid("id").primaryKey().defaultRandom(),
  companyId: uuid("company_id").notNull().references(() => companies.id),
  date: date("date").notNull(),
  open: doublePrecision("open").notNull(),
  high: doublePrecision("high").notNull(),
  low: doublePrecision("low").notNull(),
  close: doublePrecision("close").notNull(),
  adjClose: doublePrecision("adj_close"),
  volume: bigint("volume", { mode: "number" }).notNull(),
  source: text("source").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
}, (t) => [unique("uq_bar_company_date_source").on(t.companyId, t.date, t.source)]);

export const recentSearches = pgTable("recent_searches", {
  id: uuid("id").primaryKey().defaultRandom(),
  query: text("query").notNull(),
  resolvedTicker: text("resolved_ticker"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const providerState = pgTable("provider_state", {
  provider: text("provider").primaryKey(),       // "fmp" | "yahoo"
  callsToday: integer("calls_today").notNull().default(0),
  dailyLimit: integer("daily_limit").notNull(),
  resetAt: timestamp("reset_at", { withTimezone: true }).notNull(),
  lastSuccessAt: timestamp("last_success_at", { withTimezone: true }),
  lastErrorAt: timestamp("last_error_at", { withTimezone: true }),
  lastError: text("last_error"),
});
```

- [ ] **Step 2: Write the db client**

Create `lib/db/client.ts`:
```ts
import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import { env } from "@/lib/env";
import * as schema from "./schema";

const sql = neon(env.DATABASE_URL);
export const db = drizzle({ client: sql, schema });
```

- [ ] **Step 3: Write `drizzle.config.ts`**

Create `drizzle.config.ts`:
```ts
import "dotenv/config";
import { defineConfig } from "drizzle-kit";

export default defineConfig({
  out: "./drizzle",
  schema: "./lib/db/schema.ts",
  dialect: "postgresql",
  dbCredentials: { url: process.env.DATABASE_URL! },
});
```

- [ ] **Step 4: Add db scripts to `package.json`**

Add under `"scripts"`:
```json
"db:generate": "drizzle-kit generate",
"db:migrate": "drizzle-kit migrate",
"db:push": "drizzle-kit push"
```

- [ ] **Step 5: Generate and apply the migration**

Run: `pnpm db:generate && pnpm db:migrate`
Expected: a SQL file appears under `drizzle/`, and the four tables are created in Neon (no errors).

- [ ] **Step 6: Verify tables exist**

Run: `pnpm drizzle-kit push` (should report "No changes detected") — confirms schema matches DB.
Expected: "No changes detected" or an empty diff.

- [ ] **Step 7: Commit**

```bash
git add lib/db/schema.ts lib/db/client.ts drizzle.config.ts drizzle/ package.json
git commit -m "Add Drizzle schema, Neon client, and initial migration"
```

---

## Task 4: Shared types + Vitest config

**Files:**
- Create: `lib/types.ts`, `vitest.config.ts`
- Modify: `package.json` (test script)

- [ ] **Step 1: Write shared types**

Create `lib/types.ts`:
```ts
export type AssetType = "stock" | "etf" | "index";

export type PriceBar = {
  date: string;        // ISO yyyy-mm-dd
  open: number;
  high: number;
  low: number;
  close: number;
  adjClose: number | null;
  volume: number;
};

export type SearchResult = {
  symbol: string;
  name: string;
  exchange: string | null;
  assetType: AssetType;
  source: string;      // "fmp" | "yahoo" | "cache"
};

export type CompanyProfile = {
  ticker: string;
  name: string;
  assetType: AssetType;
  exchange: string | null;
  sector: string | null;
  industry: string | null;
  currency: string | null;
};

export type PeriodReturns = {
  oneDay: number | null;
  fiveDay: number | null;
  oneMonth: number | null;
  threeMonth: number | null;
  sixMonth: number | null;
  ytd: number | null;
  oneYear: number | null;
};
```

- [ ] **Step 2: Write `vitest.config.ts`**

Create `vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  plugins: [tsconfigPaths()],
  test: { environment: "node", include: ["lib/**/*.test.ts", "tests/unit/**/*.test.ts"] },
});
```

- [ ] **Step 3: Add the test script**

Add under `"scripts"` in `package.json`: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 4: Commit**

```bash
git add lib/types.ts vitest.config.ts package.json
git commit -m "Add shared types and Vitest config"
```

---

## Task 5: Indicator — period returns

**Files:**
- Create: `lib/indicators/returns.ts`
- Test: `lib/indicators/returns.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/indicators/returns.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { computeReturns } from "./returns";
import type { PriceBar } from "@/lib/types";

const bar = (date: string, close: number): PriceBar => ({
  date, open: close, high: close, low: close, close, adjClose: null, volume: 0,
});

describe("computeReturns", () => {
  it("computes 1D and 5D returns by trading-day count", () => {
    const bars: PriceBar[] = [
      bar("2025-01-02", 100), bar("2025-01-03", 101), bar("2025-01-06", 102),
      bar("2025-01-07", 103), bar("2025-01-08", 104), bar("2025-01-09", 110),
    ];
    const r = computeReturns(bars);
    expect(r.oneDay).toBeCloseTo((110 - 104) / 104, 10);   // vs previous bar
    expect(r.fiveDay).toBeCloseTo((110 - 100) / 100, 10);  // vs 5 bars back
  });

  it("computes YTD vs the last close of the previous year", () => {
    const bars: PriceBar[] = [
      bar("2024-12-31", 200), bar("2025-01-02", 210), bar("2025-01-03", 220),
    ];
    const r = computeReturns(bars);
    expect(r.ytd).toBeCloseTo((220 - 200) / 200, 10);
  });

  it("returns null when there is insufficient history", () => {
    const r = computeReturns([bar("2025-01-02", 100)]);
    expect(r.oneDay).toBeNull();
    expect(r.oneYear).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm vitest run lib/indicators/returns.test.ts`
Expected: FAIL — `computeReturns` is not defined / module not found.

- [ ] **Step 3: Write the implementation**

Create `lib/indicators/returns.ts`:
```ts
import type { PriceBar, PeriodReturns } from "@/lib/types";

function pct(curr: number, prev: number): number | null {
  if (prev === 0) return null;
  return (curr - prev) / prev;
}

function shiftMonths(iso: string, months: number): string {
  const d = new Date(iso + "T00:00:00Z");
  d.setUTCMonth(d.getUTCMonth() - months);
  return d.toISOString().slice(0, 10);
}

// bars must be sorted ascending by date
function closeOnOrBefore(bars: PriceBar[], targetIso: string): number | null {
  for (let i = bars.length - 1; i >= 0; i--) {
    if (bars[i].date <= targetIso) return bars[i].close;
  }
  return null;
}

export function computeReturns(bars: PriceBar[]): PeriodReturns {
  const empty: PeriodReturns = {
    oneDay: null, fiveDay: null, oneMonth: null, threeMonth: null,
    sixMonth: null, ytd: null, oneYear: null,
  };
  if (bars.length === 0) return empty;

  const last = bars[bars.length - 1];
  const latest = last.close;
  const latestDate = last.date;
  const latestYear = latestDate.slice(0, 4);

  const byCount = (n: number) =>
    bars.length > n ? pct(latest, bars[bars.length - 1 - n].close) : null;

  const byDate = (months: number) => {
    const ref = closeOnOrBefore(bars, shiftMonths(latestDate, months));
    return ref === null ? null : pct(latest, ref);
  };

  // YTD: last close strictly before Jan 1 of the latest year (prior year-end)
  const ytdRef = closeOnOrBefore(bars, `${Number(latestYear) - 1}-12-31`);

  return {
    oneDay: byCount(1),
    fiveDay: byCount(5),
    oneMonth: byDate(1),
    threeMonth: byDate(3),
    sixMonth: byDate(6),
    ytd: ytdRef === null ? null : pct(latest, ytdRef),
    oneYear: byDate(12),
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm vitest run lib/indicators/returns.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/indicators/returns.ts lib/indicators/returns.test.ts
git commit -m "Add period-returns indicator"
```

---

## Task 6: Indicator — moving averages

**Files:**
- Create: `lib/indicators/moving-averages.ts`
- Test: `lib/indicators/moving-averages.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/indicators/moving-averages.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { sma } from "./moving-averages";

describe("sma", () => {
  it("returns nulls until the window is full, then simple averages", () => {
    expect(sma([1, 2, 3, 4, 5], 3)).toEqual([null, null, 2, 3, 4]);
  });
  it("returns all nulls when the series is shorter than the window", () => {
    expect(sma([1, 2], 3)).toEqual([null, null]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/indicators/moving-averages.test.ts`
Expected: FAIL — `sma` not defined.

- [ ] **Step 3: Implement**

Create `lib/indicators/moving-averages.ts`:
```ts
export function sma(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    out.push(i >= period - 1 ? sum / period : null);
  }
  return out;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/indicators/moving-averages.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/indicators/moving-averages.ts lib/indicators/moving-averages.test.ts
git commit -m "Add SMA indicator"
```

---

## Task 7: Indicator — RSI (14)

**Files:**
- Create: `lib/indicators/rsi.ts`
- Test: `lib/indicators/rsi.test.ts`

Uses Wilder's smoothing. Tested via robust properties (monotonic up → 100, monotonic down → 0) plus length/edge checks.

- [ ] **Step 1: Write the failing test**

Create `lib/indicators/rsi.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { rsi } from "./rsi";

describe("rsi", () => {
  it("is 100 for a strictly increasing series (no losses)", () => {
    const values = Array.from({ length: 30 }, (_, i) => 10 + i);
    const out = rsi(values, 14);
    expect(out[out.length - 1]).toBeCloseTo(100, 6);
  });
  it("is 0 for a strictly decreasing series (no gains)", () => {
    const values = Array.from({ length: 30 }, (_, i) => 100 - i);
    const out = rsi(values, 14);
    expect(out[out.length - 1]).toBeCloseTo(0, 6);
  });
  it("returns nulls for the warm-up period and matches input length", () => {
    const values = Array.from({ length: 20 }, (_, i) => 50 + (i % 3));
    const out = rsi(values, 14);
    expect(out.length).toBe(values.length);
    expect(out[0]).toBeNull();
    expect(out[13]).toBeNull();
    expect(out[14]).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/indicators/rsi.test.ts`
Expected: FAIL — `rsi` not defined.

- [ ] **Step 3: Implement**

Create `lib/indicators/rsi.ts`:
```ts
export function rsi(values: number[], period = 14): (number | null)[] {
  const out: (number | null)[] = values.map(() => null);
  if (values.length <= period) return out;

  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const diff = values[i] - values[i - 1];
    if (diff >= 0) gainSum += diff;
    else lossSum -= diff;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;
  const toRsi = (g: number, l: number) => (l === 0 ? 100 : 100 - 100 / (1 + g / l));
  out[period] = toRsi(avgGain, avgLoss);

  for (let i = period + 1; i < values.length; i++) {
    const diff = values[i] - values[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out[i] = toRsi(avgGain, avgLoss);
  }
  return out;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/indicators/rsi.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/indicators/rsi.ts lib/indicators/rsi.test.ts
git commit -m "Add RSI indicator"
```

---

## Task 8: Indicator — MACD (12/26/9)

**Files:**
- Create: `lib/indicators/macd.ts`
- Test: `lib/indicators/macd.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/indicators/macd.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { ema, macd } from "./macd";

describe("ema", () => {
  it("equals the constant for a constant series", () => {
    const out = ema([5, 5, 5, 5, 5], 3);
    expect(out[out.length - 1]).toBeCloseTo(5, 10);
  });
});

describe("macd", () => {
  it("yields ~0 macd and signal for a constant series", () => {
    const values = Array.from({ length: 60 }, () => 42);
    const { macdLine, signalLine, histogram } = macd(values);
    const n = values.length - 1;
    expect(macdLine[n]).toBeCloseTo(0, 6);
    expect(signalLine[n]).toBeCloseTo(0, 6);
    expect(histogram[n]).toBeCloseTo(0, 6);
  });
  it("matches input length for each line", () => {
    const values = Array.from({ length: 60 }, (_, i) => 10 + Math.sin(i));
    const { macdLine, signalLine, histogram } = macd(values);
    expect(macdLine.length).toBe(values.length);
    expect(signalLine.length).toBe(values.length);
    expect(histogram.length).toBe(values.length);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/indicators/macd.test.ts`
Expected: FAIL — `ema`/`macd` not defined.

- [ ] **Step 3: Implement**

Create `lib/indicators/macd.ts`:
```ts
export function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0] ?? 0;
  for (let i = 0; i < values.length; i++) {
    prev = i === 0 ? values[0] : values[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

export function macd(values: number[], fast = 12, slow = 26, signal = 9) {
  const emaFast = ema(values, fast);
  const emaSlow = ema(values, slow);
  const macdLine = values.map((_, i) => emaFast[i] - emaSlow[i]);
  const signalLine = ema(macdLine, signal);
  const histogram = macdLine.map((v, i) => v - signalLine[i]);
  return { macdLine, signalLine, histogram };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/indicators/macd.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/indicators/macd.ts lib/indicators/macd.test.ts
git commit -m "Add MACD indicator"
```

---

## Task 9: Indicator — volatility + barrel export

**Files:**
- Create: `lib/indicators/volatility.ts`, `lib/indicators/index.ts`
- Test: `lib/indicators/volatility.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/indicators/volatility.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { rollingVolatility } from "./volatility";

describe("rollingVolatility", () => {
  it("is 0 for constant prices (zero daily returns)", () => {
    const closes = Array.from({ length: 10 }, () => 100);
    const out = rollingVolatility(closes, 5);
    expect(out[out.length - 1]).toBeCloseTo(0, 10);
  });
  it("returns null until enough returns exist", () => {
    const closes = [100, 101, 102];
    const out = rollingVolatility(closes, 5);
    expect(out[0]).toBeNull();
    expect(out[2]).toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/indicators/volatility.test.ts`
Expected: FAIL — `rollingVolatility` not defined.

- [ ] **Step 3: Implement + barrel**

Create `lib/indicators/volatility.ts`:
```ts
// Sample standard deviation of daily returns over a rolling window.
export function rollingVolatility(closes: number[], window = 5): (number | null)[] {
  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push(closes[i - 1] === 0 ? 0 : (closes[i] - closes[i - 1]) / closes[i - 1]);
  }
  // align output to closes (index 0 has no return)
  const out: (number | null)[] = closes.map(() => null);
  for (let i = window; i < closes.length; i++) {
    const slice = returns.slice(i - window, i); // window returns ending at close i
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    out[i] = Math.sqrt(variance);
  }
  return out;
}
```

Create `lib/indicators/index.ts`:
```ts
export { computeReturns } from "./returns";
export { sma } from "./moving-averages";
export { rsi } from "./rsi";
export { ema, macd } from "./macd";
export { rollingVolatility } from "./volatility";
```

- [ ] **Step 4: Run the full indicator suite**

Run: `pnpm vitest run lib/indicators`
Expected: PASS (all indicator tests).

- [ ] **Step 5: Commit**

```bash
git add lib/indicators/volatility.ts lib/indicators/volatility.test.ts lib/indicators/index.ts
git commit -m "Add volatility indicator and indicators barrel"
```

---

## Task 10: FMP provider

**Files:**
- Create: `lib/providers/base.ts`, `lib/providers/fmp.ts`
- Test: `lib/providers/fmp.test.ts`

- [ ] **Step 1: Verify the live FMP EOD response shape (one-time)**

Run (uses your key from `.env`):
```bash
set -a; source .env; set +a
curl -s "https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&from=2025-05-01&to=2025-05-09&apikey=$FMP_API_KEY" | head -c 800; echo
```
Expected: a JSON array of bars. **Confirm the exact field names** (`date`, `open`, `high`, `low`, `close`, `volume`). If casing differs from the parser in Step 4, adjust the `mapBar` function accordingly before finishing the task.

- [ ] **Step 2: Write the failing test**

Create `lib/providers/fmp.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { parseSearch, parseProfile, parseBars } from "./fmp";

describe("FMP parsers", () => {
  it("maps search results", () => {
    const raw = [{ symbol: "AAPL", name: "Apple Inc.", currency: "USD",
      exchangeFullName: "NASDAQ Global Select", exchange: "NASDAQ" }];
    expect(parseSearch(raw)).toEqual([
      { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", assetType: "stock", source: "fmp" },
    ]);
  });

  it("maps a profile and infers ETF asset type", () => {
    const raw = [{ symbol: "SPY", companyName: "SPDR S&P 500 ETF Trust", currency: "USD",
      exchange: "NYSE", sector: "", industry: "", isEtf: true, isFund: false }];
    expect(parseProfile(raw)).toEqual({
      ticker: "SPY", name: "SPDR S&P 500 ETF Trust", assetType: "etf",
      exchange: "NYSE", sector: null, industry: null, currency: "USD",
    });
  });

  it("maps EOD bars and sorts ascending by date", () => {
    const raw = [
      { date: "2025-05-02", open: 2, high: 3, low: 1, close: 2.5, volume: 100 },
      { date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, volume: 50 },
    ];
    const bars = parseBars(raw);
    expect(bars.map((b) => b.date)).toEqual(["2025-05-01", "2025-05-02"]);
    expect(bars[0]).toEqual({ date: "2025-05-01", open: 1, high: 2, low: 0.5,
      close: 1.5, adjClose: null, volume: 50 });
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `pnpm vitest run lib/providers/fmp.test.ts`
Expected: FAIL — parsers not defined.

- [ ] **Step 4: Implement provider base + FMP**

Create `lib/providers/base.ts`:
```ts
import type { PriceBar, SearchResult, CompanyProfile } from "@/lib/types";

export interface MarketDataProvider {
  name: string;
  search(query: string): Promise<SearchResult[]>;
  profile(ticker: string): Promise<CompanyProfile | null>;
  dailyPrices(ticker: string, from: string, to: string): Promise<PriceBar[]>;
}
export type { PriceBar, SearchResult, CompanyProfile };
```

Create `lib/providers/fmp.ts`:
```ts
import { env } from "@/lib/env";
import type { PriceBar, SearchResult, CompanyProfile, AssetType } from "@/lib/types";

const BASE = "https://financialmodelingprep.com/stable";

function s(v: unknown): string | null {
  return typeof v === "string" && v.trim() !== "" ? v : null;
}

export function parseSearch(raw: any[]): SearchResult[] {
  return (raw ?? []).map((r) => ({
    symbol: String(r.symbol),
    name: String(r.name ?? r.symbol),
    exchange: s(r.exchange),
    assetType: "stock" as AssetType, // search payload lacks a reliable type; refined on profile
    source: "fmp",
  }));
}

export function parseProfile(raw: any[]): CompanyProfile | null {
  const r = Array.isArray(raw) ? raw[0] : raw;
  if (!r || !r.symbol) return null;
  const assetType: AssetType = r.isEtf || r.isFund ? "etf" : "stock";
  return {
    ticker: String(r.symbol),
    name: String(r.companyName ?? r.symbol),
    assetType,
    exchange: s(r.exchange),
    sector: s(r.sector),
    industry: s(r.industry),
    currency: s(r.currency),
  };
}

export function parseBars(raw: any[]): PriceBar[] {
  return (raw ?? [])
    .map((b) => ({
      date: String(b.date).slice(0, 10),
      open: Number(b.open),
      high: Number(b.high),
      low: Number(b.low),
      close: Number(b.close),
      adjClose: null,
      volume: Number(b.volume ?? 0),
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

async function fmpGet(path: string): Promise<any> {
  const sep = path.includes("?") ? "&" : "?";
  const res = await fetch(`${BASE}${path}${sep}apikey=${env.FMP_API_KEY}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`FMP ${res.status} for ${path}`);
  return res.json();
}

export const fmp = {
  name: "fmp",
  async search(query: string): Promise<SearchResult[]> {
    const bySymbol = parseSearch(await fmpGet(`/search-symbol?query=${encodeURIComponent(query)}`));
    if (bySymbol.length > 0) return bySymbol;
    return parseSearch(await fmpGet(`/search-name?query=${encodeURIComponent(query)}`));
  },
  async profile(ticker: string): Promise<CompanyProfile | null> {
    return parseProfile(await fmpGet(`/profile?symbol=${encodeURIComponent(ticker)}`));
  },
  async dailyPrices(ticker: string, from: string, to: string): Promise<PriceBar[]> {
    return parseBars(
      await fmpGet(`/historical-price-eod/full?symbol=${encodeURIComponent(ticker)}&from=${from}&to=${to}`),
    );
  },
};
```

- [ ] **Step 5: Run to verify it passes**

Run: `pnpm vitest run lib/providers/fmp.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add lib/providers/base.ts lib/providers/fmp.ts lib/providers/fmp.test.ts
git commit -m "Add FMP provider (search, profile, daily prices)"
```

---

## Task 11: Yahoo fallback provider

**Files:**
- Create: `lib/providers/yahoo.ts`
- Test: `lib/providers/yahoo.test.ts`

- [ ] **Step 1: Write the failing test (mock yahoo-finance2)**

Create `lib/providers/yahoo.test.ts`:
```ts
import { describe, it, expect, vi } from "vitest";

const chart = vi.fn();
const search = vi.fn();
vi.mock("yahoo-finance2", () => ({
  default: class { chart = chart; search = search; },
}));

import { yahoo } from "./yahoo";

describe("yahoo provider", () => {
  it("maps chart() quotes to PriceBars (ascending, adjclose lowercase)", async () => {
    chart.mockResolvedValue({
      quotes: [
        { date: new Date("2025-05-01T00:00:00Z"), open: 1, high: 2, low: 0.5, close: 1.5, volume: 50, adjclose: 1.4 },
        { date: new Date("2025-05-02T00:00:00Z"), open: 2, high: 3, low: 1, close: 2.5, volume: 100, adjclose: 2.4 },
      ],
    });
    const bars = await yahoo.dailyPrices("AAPL", "2025-05-01", "2025-05-03");
    expect(bars).toHaveLength(2);
    expect(bars[0]).toEqual({ date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, adjClose: 1.4, volume: 50 });
  });

  it("maps search() quotes to SearchResults", async () => {
    search.mockResolvedValue({
      quotes: [{ symbol: "NVDA", exchange: "NMS", shortname: "NVIDIA Corp", typeDisp: "Equity" }],
      news: [],
    });
    const res = await yahoo.search("nvidia");
    expect(res[0]).toEqual({ symbol: "NVDA", name: "NVIDIA Corp", exchange: "NMS", assetType: "stock", source: "yahoo" });
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/providers/yahoo.test.ts`
Expected: FAIL — `yahoo` not defined.

- [ ] **Step 3: Implement**

Create `lib/providers/yahoo.ts`:
```ts
import YahooFinance from "yahoo-finance2";
import type { PriceBar, SearchResult, CompanyProfile, AssetType } from "@/lib/types";

const yf = new YahooFinance();

function toIso(d: Date | string): string {
  return (typeof d === "string" ? new Date(d) : d).toISOString().slice(0, 10);
}

export const yahoo = {
  name: "yahoo",
  async search(query: string): Promise<SearchResult[]> {
    const res: any = await yf.search(query);
    return (res.quotes ?? [])
      .filter((q: any) => q.symbol)
      .map((q: any) => {
        const t = String(q.typeDisp ?? "").toLowerCase();
        const assetType: AssetType = t === "etf" ? "etf" : t === "index" ? "index" : "stock";
        return {
          symbol: String(q.symbol),
          name: String(q.longname ?? q.shortname ?? q.symbol),
          exchange: q.exchange ?? null,
          assetType,
          source: "yahoo" as const,
        };
      });
  },
  async profile(_ticker: string): Promise<CompanyProfile | null> {
    return null; // Yahoo profile not used in SP1; FMP/cache cover profiles.
  },
  async dailyPrices(ticker: string, from: string, to: string): Promise<PriceBar[]> {
    const res: any = await yf.chart(ticker, { period1: from, period2: to, interval: "1d" });
    return (res.quotes ?? [])
      .filter((q: any) => q.close != null)
      .map((q: any) => ({
        date: toIso(q.date),
        open: Number(q.open),
        high: Number(q.high),
        low: Number(q.low),
        close: Number(q.close),
        adjClose: q.adjclose != null ? Number(q.adjclose) : null,
        volume: Number(q.volume ?? 0),
      }))
      .sort((a: PriceBar, b: PriceBar) => a.date.localeCompare(b.date));
  },
};
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/providers/yahoo.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/providers/yahoo.ts lib/providers/yahoo.test.ts
git commit -m "Add Yahoo fallback provider"
```

---

## Task 12: Provider-state quota tracking

**Files:**
- Create: `lib/db/provider-state.ts`
- Test: `tests/integration/provider-state.test.ts`

Integration tests hit the real Neon DB (via `DATABASE_URL`). Keep them isolated by using the `provider_state` table only.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/provider-state.test.ts`:
```ts
import { describe, it, expect, beforeEach } from "vitest";
import { db } from "@/lib/db/client";
import { providerState } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { canCall, recordSuccess, recordError } from "@/lib/db/provider-state";

beforeEach(async () => {
  await db.delete(providerState).where(eq(providerState.provider, "test"));
});

describe("provider-state", () => {
  it("allows calls under the daily limit and blocks at the limit", async () => {
    expect(await canCall("test", 2)).toBe(true);
    await recordSuccess("test", 2);
    expect(await canCall("test", 2)).toBe(true);
    await recordSuccess("test", 2);
    expect(await canCall("test", 2)).toBe(false); // 2/2 used
  });

  it("records errors without incrementing the call count past success calls", async () => {
    await recordError("test", 5, "boom");
    const [row] = await db.select().from(providerState).where(eq(providerState.provider, "test"));
    expect(row.lastError).toBe("boom");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run tests/integration/provider-state.test.ts`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement**

Create `lib/db/provider-state.ts`:
```ts
import { db } from "./client";
import { providerState } from "./schema";
import { eq } from "drizzle-orm";

function nextResetAt(): Date {
  const d = new Date();
  d.setUTCHours(24, 0, 0, 0); // next UTC midnight
  return d;
}

async function ensureRow(provider: string, dailyLimit: number) {
  const [row] = await db.select().from(providerState).where(eq(providerState.provider, provider));
  if (!row) {
    await db.insert(providerState).values({ provider, dailyLimit, callsToday: 0, resetAt: nextResetAt() })
      .onConflictDoNothing();
    return (await db.select().from(providerState).where(eq(providerState.provider, provider)))[0];
  }
  if (row.resetAt.getTime() <= Date.now()) {
    await db.update(providerState).set({ callsToday: 0, resetAt: nextResetAt() })
      .where(eq(providerState.provider, provider));
    return { ...row, callsToday: 0, resetAt: nextResetAt() };
  }
  return row;
}

export async function canCall(provider: string, dailyLimit: number): Promise<boolean> {
  const row = await ensureRow(provider, dailyLimit);
  return row.callsToday < dailyLimit;
}

export async function recordSuccess(provider: string, dailyLimit: number): Promise<void> {
  const row = await ensureRow(provider, dailyLimit);
  await db.update(providerState)
    .set({ callsToday: row.callsToday + 1, lastSuccessAt: new Date() })
    .where(eq(providerState.provider, provider));
}

export async function recordError(provider: string, dailyLimit: number, message: string): Promise<void> {
  await ensureRow(provider, dailyLimit);
  await db.update(providerState)
    .set({ lastErrorAt: new Date(), lastError: message })
    .where(eq(providerState.provider, provider));
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run tests/integration/provider-state.test.ts`
Expected: PASS (2 tests). (Requires `DATABASE_URL` set.)

- [ ] **Step 5: Commit**

```bash
git add lib/db/provider-state.ts tests/integration/provider-state.test.ts
git commit -m "Add provider-state quota tracking"
```

---

## Task 13: Company + price-bar + recent-search repositories

**Files:**
- Create: `lib/db/companies.ts`, `lib/db/price-bars.ts`, `lib/db/recent-searches.ts`
- Test: `tests/integration/repositories.test.ts`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/repositories.test.ts`:
```ts
import { describe, it, expect, beforeEach } from "vitest";
import { db } from "@/lib/db/client";
import { companies, priceBarsDaily } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { upsertCompany, getCompanyByTicker } from "@/lib/db/companies";
import { upsertBars, getBars } from "@/lib/db/price-bars";
import { logSearch, listRecentSearches as listRecent } from "@/lib/db/recent-searches";

async function cleanup(ticker: string) {
  const c = await getCompanyByTicker(ticker);
  if (c) {
    await db.delete(priceBarsDaily).where(eq(priceBarsDaily.companyId, c.id));
    await db.delete(companies).where(eq(companies.id, c.id));
  }
}

describe("repositories", () => {
  beforeEach(() => cleanup("TEST"));

  it("upserts a company idempotently by ticker", async () => {
    const a = await upsertCompany({ ticker: "TEST", name: "Test One", assetType: "stock",
      exchange: "NYSE", sector: null, industry: null, currency: "USD" });
    const b = await upsertCompany({ ticker: "TEST", name: "Test Two", assetType: "stock",
      exchange: "NYSE", sector: null, industry: null, currency: "USD" });
    expect(a.id).toBe(b.id);
    expect((await getCompanyByTicker("TEST"))!.name).toBe("Test Two");
  });

  it("upserts bars without duplicating on (company,date,source) and reads them ascending", async () => {
    const c = await upsertCompany({ ticker: "TEST", name: "Test", assetType: "stock",
      exchange: null, sector: null, industry: null, currency: "USD" });
    const bars = [
      { date: "2025-05-02", open: 2, high: 3, low: 1, close: 2.5, adjClose: null, volume: 100 },
      { date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, adjClose: null, volume: 50 },
    ];
    await upsertBars(c.id, bars, "fmp");
    await upsertBars(c.id, bars, "fmp"); // repeat → no dupes
    const read = await getBars(c.id);
    expect(read.map((b) => b.date)).toEqual(["2025-05-01", "2025-05-02"]);
  });

  it("logs and lists recent searches (most recent first)", async () => {
    await logSearch("nvidia", "NVDA");
    const rows = await listRecent(5);
    expect(rows[0].query).toBe("nvidia");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run tests/integration/repositories.test.ts`
Expected: FAIL — repository functions not defined.

- [ ] **Step 3: Implement the repositories**

Create `lib/db/companies.ts`:
```ts
import { db } from "./client";
import { companies } from "./schema";
import { eq } from "drizzle-orm";
import type { CompanyProfile } from "@/lib/types";

export type CompanyRow = typeof companies.$inferSelect;

export async function getCompanyByTicker(ticker: string): Promise<CompanyRow | undefined> {
  const [row] = await db.select().from(companies).where(eq(companies.ticker, ticker.toUpperCase()));
  return row;
}

export async function upsertCompany(p: CompanyProfile): Promise<CompanyRow> {
  const ticker = p.ticker.toUpperCase();
  await db.insert(companies)
    .values({ ticker, name: p.name, assetType: p.assetType, exchange: p.exchange,
      sector: p.sector, industry: p.industry, currency: p.currency,
      lastProfileRefreshAt: new Date(), updatedAt: new Date() })
    .onConflictDoUpdate({
      target: companies.ticker,
      set: { name: p.name, assetType: p.assetType, exchange: p.exchange, sector: p.sector,
        industry: p.industry, currency: p.currency, lastProfileRefreshAt: new Date(), updatedAt: new Date() },
    });
  return (await getCompanyByTicker(ticker))!;
}
```

Create `lib/db/price-bars.ts`:
```ts
import { db } from "./client";
import { priceBarsDaily } from "./schema";
import { eq, asc } from "drizzle-orm";
import type { PriceBar } from "@/lib/types";

export async function upsertBars(companyId: string, bars: PriceBar[], source: string): Promise<void> {
  if (bars.length === 0) return;
  const rows = bars.map((b) => ({
    companyId, date: b.date, open: b.open, high: b.high, low: b.low, close: b.close,
    adjClose: b.adjClose, volume: b.volume, source,
  }));
  // chunk to stay well under parameter limits
  for (let i = 0; i < rows.length; i += 500) {
    await db.insert(priceBarsDaily).values(rows.slice(i, i + 500)).onConflictDoNothing();
  }
}

export async function getBars(companyId: string): Promise<PriceBar[]> {
  const rows = await db.select().from(priceBarsDaily)
    .where(eq(priceBarsDaily.companyId, companyId)).orderBy(asc(priceBarsDaily.date));
  return rows.map((r) => ({
    date: r.date, open: r.open, high: r.high, low: r.low, close: r.close,
    adjClose: r.adjClose, volume: r.volume,
  }));
}
```

Create `lib/db/recent-searches.ts`:
```ts
import { db } from "./client";
import { recentSearches } from "./schema";
import { desc } from "drizzle-orm";

export async function logSearch(query: string, resolvedTicker: string | null): Promise<void> {
  await db.insert(recentSearches).values({ query, resolvedTicker });
}

export async function listRecentSearches(limit = 10) {
  return db.select().from(recentSearches).orderBy(desc(recentSearches.createdAt)).limit(limit);
}
```

> Note: the `recentSearches` table is imported only from `schema.ts`; the query helper is `listRecentSearches` to avoid any name collision.

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run tests/integration/repositories.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/db/companies.ts lib/db/price-bars.ts lib/db/recent-searches.ts tests/integration/repositories.test.ts
git commit -m "Add company, price-bar, and recent-search repositories"
```

---

## Task 14: Search service

**Files:**
- Create: `lib/services/search-service.ts`
- Test: `lib/services/search-service.test.ts`

- [ ] **Step 1: Write the failing test (mock providers + db logging)**

Create `lib/services/search-service.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const fmpSearch = vi.fn();
const yahooSearch = vi.fn();
const logSearch = vi.fn();
vi.mock("@/lib/providers/fmp", () => ({ fmp: { search: fmpSearch } }));
vi.mock("@/lib/providers/yahoo", () => ({ yahoo: { search: yahooSearch } }));
vi.mock("@/lib/db/recent-searches", () => ({ logSearch }));

import { resolveQuery } from "./search-service";

beforeEach(() => { fmpSearch.mockReset(); yahooSearch.mockReset(); logSearch.mockReset(); });

describe("resolveQuery", () => {
  it("returns FMP results and logs the top resolved ticker", async () => {
    fmpSearch.mockResolvedValue([{ symbol: "NVDA", name: "NVIDIA", exchange: "NASDAQ", assetType: "stock", source: "fmp" }]);
    const out = await resolveQuery("nvidia");
    expect(out[0].symbol).toBe("NVDA");
    expect(logSearch).toHaveBeenCalledWith("nvidia", "NVDA");
  });

  it("falls back to Yahoo when FMP throws, and still logs", async () => {
    fmpSearch.mockRejectedValue(new Error("fmp down"));
    yahooSearch.mockResolvedValue([{ symbol: "AMD", name: "AMD", exchange: "NMS", assetType: "stock", source: "yahoo" }]);
    const out = await resolveQuery("amd");
    expect(out[0].source).toBe("yahoo");
    expect(logSearch).toHaveBeenCalledWith("amd", "AMD");
  });

  it("logs a null ticker when nothing resolves", async () => {
    fmpSearch.mockResolvedValue([]);
    yahooSearch.mockResolvedValue([]);
    const out = await resolveQuery("zzzzz");
    expect(out).toEqual([]);
    expect(logSearch).toHaveBeenCalledWith("zzzzz", null);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/services/search-service.test.ts`
Expected: FAIL — `resolveQuery` not defined.

- [ ] **Step 3: Implement**

Create `lib/services/search-service.ts`:
```ts
import { fmp } from "@/lib/providers/fmp";
import { yahoo } from "@/lib/providers/yahoo";
import { logSearch } from "@/lib/db/recent-searches";
import type { SearchResult } from "@/lib/types";

export async function resolveQuery(query: string): Promise<SearchResult[]> {
  const q = query.trim();
  if (!q) return [];

  let results: SearchResult[] = [];
  try {
    results = await fmp.search(q);
  } catch {
    try {
      results = await yahoo.search(q);
    } catch {
      results = [];
    }
  }
  await logSearch(q, results[0]?.symbol ?? null);
  return results;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/services/search-service.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/services/search-service.ts lib/services/search-service.test.ts
git commit -m "Add search service with FMP→Yahoo fallback"
```

---

## Task 15: Price service (freshness, fallback, cache)

**Files:**
- Create: `lib/services/price-service.ts`
- Test: `lib/services/price-service.test.ts`

The service returns bars plus a `freshness` flag and `source` so the UI can show a staleness badge.

- [ ] **Step 1: Write the failing test**

Create `lib/services/price-service.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const getCompanyByTicker = vi.fn();
const upsertCompany = vi.fn();
const getBars = vi.fn();
const upsertBars = vi.fn();
const fmpProfile = vi.fn();
const fmpPrices = vi.fn();
const yahooPrices = vi.fn();
const canCall = vi.fn();
const recordSuccess = vi.fn();
const recordError = vi.fn();

vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker, upsertCompany }));
vi.mock("@/lib/db/price-bars", () => ({ getBars, upsertBars }));
vi.mock("@/lib/providers/fmp", () => ({ fmp: { profile: fmpProfile, dailyPrices: fmpPrices } }));
vi.mock("@/lib/providers/yahoo", () => ({ yahoo: { dailyPrices: yahooPrices } }));
vi.mock("@/lib/db/provider-state", () => ({ canCall, recordSuccess, recordError }));

import { getTickerData } from "./price-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", assetType: "stock" };
const freshBars = (lastDate: string) => [
  { date: "2025-05-01", open: 1, high: 1, low: 1, close: 1, adjClose: null, volume: 1 },
  { date: lastDate, open: 2, high: 2, low: 2, close: 2, adjClose: null, volume: 1 },
];

beforeEach(() => {
  [getCompanyByTicker, upsertCompany, getBars, upsertBars, fmpProfile, fmpPrices, yahooPrices,
    canCall, recordSuccess, recordError].forEach((m) => m.mockReset());
  upsertCompany.mockResolvedValue(company);
  getCompanyByTicker.mockResolvedValue(company);
  canCall.mockResolvedValue(true);
});

describe("getTickerData", () => {
  it("serves cache without calling a provider when bars are fresh", async () => {
    const today = new Date().toISOString().slice(0, 10);
    getBars.mockResolvedValue(freshBars(today));
    const out = await getTickerData("NVDA", "1y");
    expect(fmpPrices).not.toHaveBeenCalled();
    expect(out.source).toBe("cache");
    expect(out.bars.length).toBe(2);
  });

  it("fetches from FMP when cache is stale, then upserts and serves", async () => {
    getBars.mockResolvedValueOnce([]).mockResolvedValueOnce(freshBars("2020-01-01"));
    fmpProfile.mockResolvedValue({ ticker: "NVDA", name: "NVIDIA", assetType: "stock",
      exchange: "NASDAQ", sector: null, industry: null, currency: "USD" });
    fmpPrices.mockResolvedValue(freshBars("2020-01-01"));
    const out = await getTickerData("NVDA", "1y");
    expect(fmpPrices).toHaveBeenCalled();
    expect(upsertBars).toHaveBeenCalledWith("c1", expect.any(Array), "fmp");
    expect(recordSuccess).toHaveBeenCalled();
    expect(out.source).toBe("fmp");
  });

  it("falls back to Yahoo when FMP fails, marking the source", async () => {
    getBars.mockResolvedValueOnce([]).mockResolvedValueOnce(freshBars("2020-01-01"));
    fmpProfile.mockResolvedValue(null);
    fmpPrices.mockRejectedValue(new Error("fmp 429"));
    yahooPrices.mockResolvedValue(freshBars("2020-01-01"));
    const out = await getTickerData("NVDA", "1y");
    expect(recordError).toHaveBeenCalled();
    expect(upsertBars).toHaveBeenCalledWith("c1", expect.any(Array), "yahoo");
    expect(out.source).toBe("yahoo");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/services/price-service.test.ts`
Expected: FAIL — `getTickerData` not defined.

- [ ] **Step 3: Implement**

Create `lib/services/price-service.ts`:
```ts
import { env } from "@/lib/env";
import { fmp } from "@/lib/providers/fmp";
import { yahoo } from "@/lib/providers/yahoo";
import { getCompanyByTicker, upsertCompany } from "@/lib/db/companies";
import { getBars, upsertBars } from "@/lib/db/price-bars";
import { canCall, recordSuccess, recordError } from "@/lib/db/provider-state";
import { computeReturns, sma, rsi, macd, rollingVolatility } from "@/lib/indicators";
import type { PriceBar, PeriodReturns } from "@/lib/types";

export type Range = "1m" | "3m" | "6m" | "ytd" | "1y";

export type TickerData = {
  ticker: string;
  bars: PriceBar[];
  returns: PeriodReturns;
  indicators: {
    ma10: (number | null)[]; ma20: (number | null)[]; ma50: (number | null)[];
    rsi14: (number | null)[]; macdHistogram: number[]; volatility5d: (number | null)[];
  };
  source: "cache" | "fmp" | "yahoo";
  lastBarDate: string | null;
  stale: boolean;
};

function lastTradingDayIso(): string {
  const d = new Date();
  const day = d.getUTCDay();           // 0 Sun .. 6 Sat
  if (day === 0) d.setUTCDate(d.getUTCDate() - 2);
  else if (day === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

function isFresh(bars: PriceBar[]): boolean {
  if (bars.length === 0) return false;
  return bars[bars.length - 1].date >= lastTradingDayIso();
}

async function ensureCompany(ticker: string) {
  const existing = await getCompanyByTicker(ticker);
  if (existing) return existing;
  const profile = await fmp.profile(ticker).catch(() => null);
  return upsertCompany(
    profile ?? { ticker, name: ticker, assetType: "stock", exchange: null, sector: null, industry: null, currency: null },
  );
}

export async function getTickerData(ticker: string, _range: Range = "1y"): Promise<TickerData> {
  const company = await ensureCompany(ticker);
  let bars = await getBars(company.id);
  let source: TickerData["source"] = "cache";

  if (!isFresh(bars)) {
    const from = new Date(); from.setUTCFullYear(from.getUTCFullYear() - 2);
    const fromIso = from.toISOString().slice(0, 10);
    const toIso = new Date().toISOString().slice(0, 10);

    let fetched: PriceBar[] | null = null;
    if (await canCall("fmp", env.FMP_DAILY_LIMIT)) {
      try {
        fetched = await fmp.dailyPrices(company.ticker, fromIso, toIso);
        await recordSuccess("fmp", env.FMP_DAILY_LIMIT);
        if (fetched.length) { await upsertBars(company.id, fetched, "fmp"); source = "fmp"; }
      } catch (e) {
        await recordError("fmp", env.FMP_DAILY_LIMIT, String(e));
        fetched = null;
      }
    }
    if (fetched === null || fetched.length === 0) {
      try {
        const y = await yahoo.dailyPrices(company.ticker, fromIso, toIso);
        if (y.length) { await upsertBars(company.id, y, "yahoo"); source = "yahoo"; }
      } catch { /* degrade to whatever cache we have */ }
    }
    bars = await getBars(company.id);
  }

  const closes = bars.map((b) => b.close);
  const lastBarDate = bars.at(-1)?.date ?? null;
  return {
    ticker: company.ticker,
    bars,
    returns: computeReturns(bars),
    indicators: {
      ma10: sma(closes, 10), ma20: sma(closes, 20), ma50: sma(closes, 50),
      rsi14: rsi(closes, 14), macdHistogram: macd(closes).histogram,
      volatility5d: rollingVolatility(closes, 5),
    },
    source,
    lastBarDate,
    stale: lastBarDate !== null && lastBarDate < lastTradingDayIso(),
  };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/services/price-service.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/services/price-service.ts lib/services/price-service.test.ts
git commit -m "Add price service with freshness, FMP→Yahoo fallback, and indicators"
```

---

## Task 16: Comparison service

**Files:**
- Create: `lib/services/comparison-service.ts`
- Test: `lib/services/comparison-service.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/services/comparison-service.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const getTickerData = vi.fn();
vi.mock("./price-service", () => ({ getTickerData }));

import { compareTickers } from "./comparison-service";
import type { PriceBar } from "@/lib/types";

const bars = (closes: [string, number][]): PriceBar[] =>
  closes.map(([date, c]) => ({ date, open: c, high: c, low: c, close: c, adjClose: null, volume: 0 }));

beforeEach(() => getTickerData.mockReset());

describe("compareTickers", () => {
  it("aligns on common dates and computes normalized series + relative return", async () => {
    getTickerData.mockImplementation((t: string) => ({
      ticker: t,
      bars: t === "A"
        ? bars([["2025-01-02", 100], ["2025-01-03", 110], ["2025-01-06", 120]])
        : bars([["2025-01-03", 50], ["2025-01-06", 55]]), // no 01-02
    }));
    const out = await compareTickers("A", "B", "1y");
    // common dates = 01-03, 01-06
    expect(out.dates).toEqual(["2025-01-03", "2025-01-06"]);
    // normalized to 100 at first common date
    expect(out.primary.normalized[0]).toBeCloseTo(100, 6);
    expect(out.primary.normalized[1]).toBeCloseTo((120 / 110) * 100, 6);
    expect(out.comparison.normalized[1]).toBeCloseTo((55 / 50) * 100, 6);
    // relative return over the window = primaryReturn - comparisonReturn
    expect(out.relativeReturn).toBeCloseTo((120 / 110 - 1) - (55 / 50 - 1), 6);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/services/comparison-service.test.ts`
Expected: FAIL — `compareTickers` not defined.

- [ ] **Step 3: Implement**

Create `lib/services/comparison-service.ts`:
```ts
import { getTickerData, type Range } from "./price-service";
import { rollingVolatility } from "@/lib/indicators";
import type { PriceBar } from "@/lib/types";

export type CompareSeries = { ticker: string; normalized: number[]; volatility: number | null; maxDrawdown: number };
export type ComparisonResult = {
  dates: string[];
  primary: CompareSeries;
  comparison: CompareSeries;
  relativeReturn: number | null;
};

function closeByDate(bars: PriceBar[]): Map<string, number> {
  return new Map(bars.map((b) => [b.date, b.close]));
}

function maxDrawdown(values: number[]): number {
  let peak = -Infinity, mdd = 0;
  for (const v of values) {
    peak = Math.max(peak, v);
    if (peak > 0) mdd = Math.min(mdd, v / peak - 1);
  }
  return mdd; // <= 0
}

export async function compareTickers(primary: string, comparison: string, range: Range): Promise<ComparisonResult> {
  const [a, b] = await Promise.all([getTickerData(primary, range), getTickerData(comparison, range)]);
  const ma = closeByDate(a.bars), mb = closeByDate(b.bars);
  const dates = [...ma.keys()].filter((d) => mb.has(d)).sort();

  const aCloses = dates.map((d) => ma.get(d)!);
  const bCloses = dates.map((d) => mb.get(d)!);

  const norm = (xs: number[]) => (xs.length ? xs.map((x) => (x / xs[0]) * 100) : []);
  const lastVol = (xs: number[]) => rollingVolatility(xs, 5).at(-1) ?? null;
  const windowReturn = (xs: number[]) => (xs.length >= 2 ? xs.at(-1)! / xs[0] - 1 : null);

  const aRet = windowReturn(aCloses);
  const bRet = windowReturn(bCloses);

  return {
    dates,
    primary: { ticker: a.ticker, normalized: norm(aCloses), volatility: lastVol(aCloses), maxDrawdown: maxDrawdown(aCloses) },
    comparison: { ticker: b.ticker, normalized: norm(bCloses), volatility: lastVol(bCloses), maxDrawdown: maxDrawdown(bCloses) },
    relativeReturn: aRet !== null && bRet !== null ? aRet - bRet : null,
  };
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/services/comparison-service.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add lib/services/comparison-service.ts lib/services/comparison-service.test.ts
git commit -m "Add comparison service (alignment, normalization, drawdown)"
```

---

## Task 17: API routes

**Files:**
- Create: `app/api/search/route.ts`, `app/api/prices/[symbol]/route.ts`, `app/api/compare/route.ts`

These are thin wrappers with Zod validation; the services are already unit-tested, so we verify the routes with manual `curl` checks against the running dev server.

- [ ] **Step 1: Implement the search route**

Create `app/api/search/route.ts`:
```ts
import { type NextRequest } from "next/server";
import { z } from "zod";
import { resolveQuery } from "@/lib/services/search-service";

const Query = z.object({ q: z.string().min(1).max(64) });

export async function GET(request: NextRequest) {
  const parsed = Query.safeParse({ q: request.nextUrl.searchParams.get("q") ?? "" });
  if (!parsed.success) return Response.json({ error: "Invalid query" }, { status: 400 });
  try {
    return Response.json({ results: await resolveQuery(parsed.data.q) });
  } catch (e) {
    return Response.json({ error: "Search failed", detail: String(e) }, { status: 502 });
  }
}
```

- [ ] **Step 2: Implement the prices route (Next 15 awaited params)**

Create `app/api/prices/[symbol]/route.ts`:
```ts
import { type NextRequest } from "next/server";
import { z } from "zod";
import { getTickerData } from "@/lib/services/price-service";

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);
const Range = z.enum(["1m", "3m", "6m", "ytd", "1y"]).default("1y");

export async function GET(request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  const range = Range.parse(request.nextUrl.searchParams.get("range") ?? undefined);
  try {
    return Response.json(await getTickerData(t.data.toUpperCase(), range));
  } catch (e) {
    return Response.json({ error: "Price lookup failed", detail: String(e) }, { status: 502 });
  }
}
```

- [ ] **Step 3: Implement the compare route**

Create `app/api/compare/route.ts`:
```ts
import { type NextRequest } from "next/server";
import { z } from "zod";
import { compareTickers } from "@/lib/services/comparison-service";

const Schema = z.object({
  primary: z.string().regex(/^[A-Za-z.\-]{1,10}$/),
  comparison: z.string().regex(/^[A-Za-z.\-]{1,10}$/),
  range: z.enum(["1m", "3m", "6m", "ytd", "1y"]).default("1y"),
});

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const parsed = Schema.safeParse({
    primary: sp.get("primary"), comparison: sp.get("comparison"), range: sp.get("range") ?? undefined,
  });
  if (!parsed.success) return Response.json({ error: "Invalid params" }, { status: 400 });
  try {
    const { primary, comparison, range } = parsed.data;
    return Response.json(await compareTickers(primary.toUpperCase(), comparison.toUpperCase(), range));
  } catch (e) {
    return Response.json({ error: "Compare failed", detail: String(e) }, { status: 502 });
  }
}
```

- [ ] **Step 4: Verify the routes against the dev server**

Run in one terminal: `pnpm dev`. In another:
```bash
curl -s "http://localhost:3000/api/search?q=nvidia" | head -c 300; echo
curl -s "http://localhost:3000/api/prices/NVDA?range=1y" | head -c 300; echo
curl -s "http://localhost:3000/api/compare?primary=NVDA&comparison=AMD&range=1y" | head -c 300; echo
```
Expected: search returns a `results` array containing NVDA; prices returns a payload with `bars`/`returns`/`source`; compare returns `dates`/`primary`/`comparison`. (First calls hit FMP and populate the cache.)

- [ ] **Step 5: Commit**

```bash
git add app/api
git commit -m "Add search, prices, and compare API routes with Zod validation"
```

---

## Task 18: App shell + TanStack Query provider + formatters

**Files:**
- Create: `app/providers.tsx`, `lib/formatters.ts`, `components/staleness-badge.tsx`
- Modify: `app/layout.tsx`

- [ ] **Step 1: Write formatter unit tests**

Create `lib/formatters.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { formatPercent, formatPrice } from "./formatters";

describe("formatters", () => {
  it("formats decimal fractions as signed percentages", () => {
    expect(formatPercent(0.0532)).toBe("+5.32%");
    expect(formatPercent(-0.01)).toBe("-1.00%");
    expect(formatPercent(null)).toBe("—");
  });
  it("formats prices with two decimals", () => {
    expect(formatPrice(262.8)).toBe("262.80");
    expect(formatPrice(null)).toBe("—");
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `pnpm vitest run lib/formatters.test.ts`
Expected: FAIL — formatters not defined.

- [ ] **Step 3: Implement formatters**

Create `lib/formatters.ts`:
```ts
export function formatPercent(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  const pct = v * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}
export function formatPrice(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v.toFixed(2);
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pnpm vitest run lib/formatters.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the Query provider and wire the layout**

Create `app/providers.tsx`:
```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

Edit `app/layout.tsx` to wrap children with `<Providers>` and set the dark, dense base. Replace the body content:
```tsx
import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = { title: "Finance Dashboard", description: "Investing research" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

Create `components/staleness-badge.tsx`:
```tsx
export function StalenessBadge({ stale, lastBarDate, source }: { stale: boolean; lastBarDate: string | null; source: string }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs ${stale ? "bg-amber-900 text-amber-200" : "bg-neutral-800 text-neutral-400"}`}>
      {lastBarDate ? `data as of ${lastBarDate}` : "no data"} · {source}{stale ? " · stale" : ""}
    </span>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add app/providers.tsx app/layout.tsx lib/formatters.ts lib/formatters.test.ts components/staleness-badge.tsx
git commit -m "Add app shell, Query provider, formatters, and staleness badge"
```

---

## Task 19: Homepage — search bar + recent searches

**Files:**
- Create: `components/search-bar.tsx`, `components/recent-searches.tsx`
- Modify: `app/page.tsx`

- [ ] **Step 1: Implement the search bar (client component)**

Create `components/search-bar.tsx`:
```tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import type { SearchResult } from "@/lib/types";

export function SearchBar() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      setResults(data.results ?? []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-xl">
      <form onSubmit={run} className="flex gap-2">
        <input
          value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="Search ticker or company (e.g. NVDA, NVIDIA)"
          className="flex-1 rounded bg-neutral-900 px-3 py-2 outline-none ring-1 ring-neutral-800 focus:ring-neutral-600"
        />
        <button className="rounded bg-neutral-200 px-4 py-2 font-medium text-neutral-900" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>
      {results.length > 0 && (
        <ul className="mt-2 divide-y divide-neutral-800 rounded bg-neutral-900 ring-1 ring-neutral-800">
          {results.slice(0, 8).map((r) => (
            <li key={`${r.symbol}-${r.source}`}>
              <button
                onClick={() => router.push(`/ticker/${r.symbol}`)}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-neutral-800"
              >
                <span><span className="font-mono font-semibold">{r.symbol}</span> · {r.name}</span>
                <span className="text-xs text-neutral-500">{r.exchange ?? ""}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Implement recent searches (server component reading the DB)**

Create `components/recent-searches.tsx`:
```tsx
import Link from "next/link";
import { listRecentSearches } from "@/lib/db/recent-searches";

export async function RecentSearches() {
  const rows = await listRecentSearches(8);
  const resolved = rows.filter((r) => r.resolvedTicker);
  if (resolved.length === 0) return null;
  return (
    <div className="mt-8 w-full max-w-xl">
      <h2 className="mb-2 text-xs uppercase tracking-wide text-neutral-500">Recent</h2>
      <div className="flex flex-wrap gap-2">
        {resolved.map((r) => (
          <Link key={r.id} href={`/ticker/${r.resolvedTicker}`}
            className="rounded bg-neutral-900 px-3 py-1 font-mono text-sm ring-1 ring-neutral-800 hover:bg-neutral-800">
            {r.resolvedTicker}
          </Link>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire the homepage**

Replace `app/page.tsx`:
```tsx
import { SearchBar } from "@/components/search-bar";
import { RecentSearches } from "@/components/recent-searches";

export const dynamic = "force-dynamic"; // recent searches read the DB per request

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center px-4 pt-24">
      <h1 className="mb-8 text-2xl font-semibold">Finance Dashboard</h1>
      <SearchBar />
      <RecentSearches />
    </main>
  );
}
```

- [ ] **Step 4: Verify in the browser**

Run: `pnpm dev`, open http://localhost:3000, search "NVIDIA", confirm NVDA appears in results and clicking navigates to `/ticker/NVDA` (page will 404 until Task 20). Recent chip appears after a search.

- [ ] **Step 5: Commit**

```bash
git add components/search-bar.tsx components/recent-searches.tsx app/page.tsx
git commit -m "Add search-first homepage with recent searches"
```

---

## Task 20: Ticker research page

**Files:**
- Create: `components/price-chart.tsx`, `components/returns-table.tsx`, `app/ticker/[symbol]/page.tsx`

- [ ] **Step 1: Implement the returns table**

Create `components/returns-table.tsx`:
```tsx
import { formatPercent } from "@/lib/formatters";
import type { PeriodReturns } from "@/lib/types";

const LABELS: [keyof PeriodReturns, string][] = [
  ["oneDay", "1D"], ["fiveDay", "5D"], ["oneMonth", "1M"],
  ["threeMonth", "3M"], ["sixMonth", "6M"], ["ytd", "YTD"], ["oneYear", "1Y"],
];

export function ReturnsTable({ returns }: { returns: PeriodReturns }) {
  return (
    <div className="grid grid-cols-7 gap-px overflow-hidden rounded ring-1 ring-neutral-800">
      {LABELS.map(([key, label]) => {
        const v = returns[key];
        const color = v === null ? "text-neutral-500" : v >= 0 ? "text-emerald-400" : "text-red-400";
        return (
          <div key={key} className="bg-neutral-900 px-2 py-2 text-center">
            <div className="text-[10px] uppercase text-neutral-500">{label}</div>
            <div className={`text-sm font-medium ${color}`}>{formatPercent(v)}</div>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 2: Implement the price chart (Recharts, line + MA overlays)**

Create `components/price-chart.tsx`:
```tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; ma20: (number | null)[]; ma50: (number | null)[] };

export function PriceChart({ bars, ma20, ma50 }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, close: b.close, ma20: ma20[i], ma50: ma50[i] }));
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Line type="monotone" dataKey="close" stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="ma20" stroke="#38bdf8" dot={false} strokeWidth={1} />
          <Line type="monotone" dataKey="ma50" stroke="#f59e0b" dot={false} strokeWidth={1} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Implement the ticker page (server component)**

Create `app/ticker/[symbol]/page.tsx`:
```tsx
import Link from "next/link";
import { getTickerData } from "@/lib/services/price-service";
import { ReturnsTable } from "@/components/returns-table";
import { PriceChart } from "@/components/price-chart";
import { StalenessBadge } from "@/components/staleness-badge";
import { formatPrice } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function TickerPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const data = await getTickerData(symbol.toUpperCase(), "1y");
  const latest = data.bars.at(-1)?.close ?? null;

  if (data.bars.length === 0) {
    return (
      <main className="mx-auto max-w-4xl px-4 pt-16">
        <Link href="/" className="text-sm text-neutral-500">← Search</Link>
        <p className="mt-8">Couldn’t resolve <span className="font-mono">{symbol.toUpperCase()}</span>. Try another ticker.</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href="/" className="text-sm text-neutral-500">← Search</Link>
      <div className="mt-4 flex items-baseline justify-between">
        <div>
          <h1 className="font-mono text-3xl font-semibold">{data.ticker}</h1>
          <div className="text-2xl">{formatPrice(latest)}</div>
        </div>
        <StalenessBadge stale={data.stale} lastBarDate={data.lastBarDate} source={data.source} />
      </div>

      <div className="mt-4"><ReturnsTable returns={data.returns} /></div>
      <div className="mt-6"><PriceChart bars={data.bars} ma20={data.indicators.ma20} ma50={data.indicators.ma50} /></div>

      <form action="/compare" className="mt-8 flex items-center gap-2">
        <input type="hidden" name="primary" value={data.ticker} />
        <label className="text-sm text-neutral-400">Compare against</label>
        <input name="comparison" placeholder="e.g. SPY"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900">Go</button>
      </form>
    </main>
  );
}
```

- [ ] **Step 4: Verify in the browser**

Run: `pnpm dev`, open http://localhost:3000/ticker/NVDA. Confirm: header price, the 7 return cells, a price chart with two MA lines, a staleness badge, and a comparison input. Try an invalid ticker (`/ticker/ZZZZZZ`) → friendly "couldn't resolve" message, no crash.

- [ ] **Step 5: Commit**

```bash
git add components/price-chart.tsx components/returns-table.tsx app/ticker
git commit -m "Add ticker research page (header, returns, price chart, compare entry)"
```

---

## Task 21: Compare page

**Files:**
- Create: `components/comparison-chart.tsx`, `app/compare/page.tsx`

- [ ] **Step 1: Implement the comparison chart**

Create `components/comparison-chart.tsx`:
```tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";

type Props = { dates: string[]; primaryTicker: string; comparisonTicker: string; primary: number[]; comparison: number[] };

export function ComparisonChart({ dates, primaryTicker, comparisonTicker, primary, comparison }: Props) {
  const data = dates.map((date, i) => ({ date, [primaryTicker]: primary[i], [comparisonTicker]: comparison[i] }));
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey={primaryTicker} stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey={comparisonTicker} stroke="#38bdf8" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Implement the compare page (reads query params)**

Create `app/compare/page.tsx`:
```tsx
import Link from "next/link";
import { compareTickers } from "@/lib/services/comparison-service";
import { ComparisonChart } from "@/components/comparison-chart";
import { formatPercent } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function ComparePage({
  searchParams,
}: { searchParams: Promise<{ primary?: string; comparison?: string; range?: string }> }) {
  const sp = await searchParams;
  const primary = (sp.primary ?? "").toUpperCase();
  const comparison = (sp.comparison ?? "").toUpperCase();

  if (!primary || !comparison) {
    return <main className="mx-auto max-w-4xl px-4 pt-16"><p>Provide both a primary and comparison ticker.</p></main>;
  }

  const r = await compareTickers(primary, comparison, "1y");
  const noOverlap = r.dates.length === 0;

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href={`/ticker/${primary}`} className="text-sm text-neutral-500">← {primary}</Link>
      <h1 className="mt-4 text-2xl font-semibold">
        <span className="font-mono">{primary}</span> vs <span className="font-mono">{comparison}</span>
      </h1>

      {noOverlap ? (
        <p className="mt-6 text-neutral-400">No overlapping price history to compare.</p>
      ) : (
        <>
          <p className="mt-2 text-sm text-neutral-400">
            Relative return (1Y window): <span className={r.relativeReturn! >= 0 ? "text-emerald-400" : "text-red-400"}>
              {formatPercent(r.relativeReturn)}</span>
          </p>
          <div className="mt-6">
            <ComparisonChart dates={r.dates} primaryTicker={primary} comparisonTicker={comparison}
              primary={r.primary.normalized} comparison={r.comparison.normalized} />
          </div>
          <table className="mt-8 w-full text-sm">
            <thead><tr className="text-left text-neutral-500">
              <th className="py-1">Metric</th><th>{primary}</th><th>{comparison}</th></tr></thead>
            <tbody>
              <tr><td className="py-1 text-neutral-400">5D volatility</td>
                <td>{formatPercent(r.primary.volatility)}</td><td>{formatPercent(r.comparison.volatility)}</td></tr>
              <tr><td className="py-1 text-neutral-400">Max drawdown</td>
                <td className="text-red-400">{formatPercent(r.primary.maxDrawdown)}</td>
                <td className="text-red-400">{formatPercent(r.comparison.maxDrawdown)}</td></tr>
            </tbody>
          </table>
        </>
      )}
    </main>
  );
}
```

- [ ] **Step 3: Verify in the browser**

Run: `pnpm dev`, open http://localhost:3000/compare?primary=NVDA&comparison=AMD and also `?primary=AAPL&comparison=SPY`. Confirm: overlay chart with both normalized series, relative-return line, and the volatility/drawdown table. Confirm an unknown comparison ticker degrades to the "no overlapping history" message rather than crashing.

- [ ] **Step 4: Commit**

```bash
git add components/comparison-chart.tsx app/compare
git commit -m "Add compare page (overlay chart, relative return, volatility/drawdown table)"
```

---

## Task 22: E2E smoke test (Playwright)

**Files:**
- Create: `playwright.config.ts`, `tests/e2e/smoke.spec.ts`
- Modify: `package.json` (script)

- [ ] **Step 1: Install browsers and add config**

Run: `pnpm exec playwright install chromium`

Create `playwright.config.ts`:
```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:3000" },
  webServer: { command: "pnpm dev", url: "http://localhost:3000", reuseExistingServer: true, timeout: 120_000 },
});
```

- [ ] **Step 2: Write the smoke test**

Create `tests/e2e/smoke.spec.ts`:
```ts
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
```

- [ ] **Step 3: Add the script and run**

Add under `"scripts"`: `"test:e2e": "playwright test"`.

Run: `pnpm test:e2e`
Expected: both tests PASS (requires `.env` with `FMP_API_KEY` + `DATABASE_URL`; first run populates the cache).

- [ ] **Step 4: Commit**

```bash
git add playwright.config.ts tests/e2e/smoke.spec.ts package.json
git commit -m "Add Playwright E2E smoke tests"
```

---

## Task 23: Full verification pass

**Files:** none (verification)

- [ ] **Step 1: Run the full unit/integration suite**

Run: `pnpm test`
Expected: all Vitest suites pass (indicators, providers, services, formatters, integration). Requires `DATABASE_URL`.

- [ ] **Step 2: Typecheck and lint**

Run: `pnpm exec tsc --noEmit && pnpm lint`
Expected: no type errors, no lint errors.

- [ ] **Step 3: Confirm SP1 acceptance criteria (manual, in browser)**

With `pnpm dev` running, verify against the spec's SP1 acceptance criteria:
- Search "NVIDIA" → NVDA appears; "NVDA" opens the ticker page.
- Ticker page shows price chart, the 7 period returns, and MA overlays.
- `NVDA` vs `AMD` and `AAPL` vs `SPY` comparisons render (overlay + relative return + volatility + drawdown).
- A provider failure / exhausted budget still shows cached data with a staleness badge (simulate by temporarily setting `FMP_DAILY_LIMIT=0` in `.env`, reloading a cached ticker, and confirming it serves cache without crashing — then restore `FMP_DAILY_LIMIT=250`).

- [ ] **Step 4: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "SP1 verification fixes"
```

---

## Self-review notes (resolved during planning)

- **Spec coverage:** search (Tasks 14,17,19), prices+freshness+fallback (Tasks 10,11,15,17,20), comparison (Tasks 16,17,21), four cache tables (Task 3), indicators computed on the fly (Tasks 5–9,15), `provider_state` budget tracking (Tasks 12,15), recent-searches-only/no watchlist (Tasks 13,19), error handling/degradation + staleness badge (Tasks 15,18,20,21,23), testing strategy (unit + integration + E2E across the plan). All SP1 acceptance criteria map to Task 23.
- **Deferred to later SPs (intentionally absent):** news/sentiment/memo (SP2), SEC fundamentals + `cik` column (SP3), password gate + provider-health page + retention pruning + Vercel/Neon prod deploy (SP4).
- **Type consistency:** `PriceBar`, `SearchResult`, `CompanyProfile`, `PeriodReturns`, `AssetType` defined once in `lib/types.ts` and reused; service return types (`TickerData`, `ComparisonResult`) defined where produced and consumed by their routes/pages.
- **Known follow-up (built into the plan):** FMP historical field casing is verified live in Task 10 Step 1 before the parser is trusted; Yahoo `adjclose` lowercase handled in Task 11.
