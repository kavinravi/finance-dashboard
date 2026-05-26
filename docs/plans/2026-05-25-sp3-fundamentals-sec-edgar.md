# SP3 — Fundamentals (SEC EDGAR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SEC EDGAR–sourced fundamentals card to the stock ticker page — official-filing values (valuation, profitability, financial health, latest financials), cached, every figure traceable to a 10-K.

**Architecture:** One `companyfacts` fetch per company (cold cache only), normalized by pure extract/derive functions, cached as a small per-company snapshot in Postgres. The card is client-fetched from `/api/fundamentals/[symbol]` (mirrors the SP2 memo card) so a cold SEC fetch never blocks page paint. Latest-annual (10-K) flow figures + latest balance sheet; price-derived multiples computed at request time from the already-cached close.

**Tech Stack:** Next.js 16 (App Router), TypeScript, Drizzle + Neon Postgres, Zod, Vitest, Playwright. SEC EDGAR JSON APIs (`company_tickers.json`, `companyfacts`).

**Spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §8.

**Standing constraints:**
- No AI-authorship traces in commits/docs/branches (no "Claude"/"Anthropic"/"Co-Authored-By"/tool names). Git author `kavinravi` is correct.
- `gemini-3.5-flash` is the verified 2026 stable model — unrelated here, but don't let any reviewer "fix" it elsewhere.
- Do NOT run `pnpm lint`/eslint locally (OOM-crashes); lint is verified in CI.
- Tests: `pnpm test` (Vitest; integration hits live Neon via `dotenv/config`), `pnpm test:e2e` (Playwright). Migrations: `pnpm db:generate` then `pnpm db:migrate`.
- After the build, do a **live SEC smoke** (real fetch for NVDA + AAPL) + a screenshot before declaring done — mocks can't catch real concept-tagging quirks.

**SEC verified 2026-05-25:** `https://www.sec.gov/files/company_tickers.json` maps index→`{cik_str:int, ticker, title}` (NVDA=1045810, AAPL=320193). `data.sec.gov` requires a contact `User-Agent` (403 without). `companyfacts` shape: `{cik, entityName, facts: {"us-gaap": {Concept: {units: {"USD": [{start,end,val,accn,fy,fp,form,filed,frame}]}}}, "dei": {...}}}`. AAPL facts = 3.75 MB / 503 us-gaap concepts; all needed concepts present & current.

---

## File Structure

**Create:**
- `lib/providers/sec.ts` — SEC HTTP: `findCik` (pure), `sec.resolveCik`, `sec.fetchCompanyFacts`.
- `lib/providers/sec.test.ts` — `findCik` unit tests.
- `lib/fundamentals/extract.ts` — pure `extractConcepts(raw)` → values + provenance meta.
- `lib/fundamentals/extract.test.ts` — extraction unit tests (inline fixtures).
- `lib/fundamentals/derive.ts` — pure `deriveMetrics(concepts, latestClose)` → display view.
- `lib/fundamentals/derive.test.ts` — derivation unit tests.
- `lib/db/fundamentals.ts` — snapshot repo (`getFundamentalsSnapshot`, `upsertFundamentalsSnapshot`).
- `lib/services/fundamentals-service.ts` — orchestration (cache-first → CIK → fetch → extract → store → derive).
- `lib/services/fundamentals-service.test.ts` — integration (live Neon, SEC mocked).
- `app/api/fundamentals/[symbol]/route.ts` — client-fetched endpoint.
- `components/fundamentals-card.tsx` — client card.

**Modify:**
- `lib/env.ts` — add `SEC_USER_AGENT`.
- `.env.example` — document `SEC_USER_AGENT`.
- `lib/types.ts` — add `SecFact`, `RawCompanyFacts`, `FundamentalConcepts`, `FundamentalsMeta`, `ExtractedFundamentals`, `FundamentalsView`.
- `lib/db/schema.ts` — add `cik` to `companies`; add `companyFundamentals` table.
- `lib/db/companies.ts` — add `setCik`.
- `lib/formatters.ts` — add `formatLargeCurrency`, `formatMultiple`, `formatRatioPercent`.
- `app/ticker/[symbol]/page.tsx` — render `<FundamentalsCard>` between memo and news.
- `tests/e2e/smoke.spec.ts` — assert the fundamentals card renders (intercept `/api/fundamentals`).
- `drizzle/` — generated migration (`0002_*.sql`).

---

## Task 1: Env + `.env.example` (`SEC_USER_AGENT`)

**Files:**
- Modify: `lib/env.ts`
- Modify: `.env.example`

- [ ] **Step 1: Add `SEC_USER_AGENT` to the schema and the parsed object in `lib/env.ts`**

In the `z.object({...})` schema add (after `GEMINI_DAILY_LIMIT`):

```ts
  SEC_USER_AGENT: z.string().min(1).optional(), // SEC requires a contact UA; feature degrades if absent
```

In the `schema.safeParse({...})` object add:

```ts
  SEC_USER_AGENT: process.env.SEC_USER_AGENT,
```

- [ ] **Step 2: Document it in `.env.example`**

Append:

```
# SEC EDGAR requires a contact User-Agent header (e.g. "AppName your@email.com").
# Without it, SEC returns 403 and the fundamentals card shows "unavailable".
SEC_USER_AGENT="finance-dashboard your@email.com"
```

- [ ] **Step 3: Set a real value in the gitignored `.env`** so live tests work:

```
SEC_USER_AGENT="finance-dashboard kavinravi121@gmail.com"
```

- [ ] **Step 4: Verify env still loads**

Run: `pnpm exec tsc --noEmit`
Expected: PASS (no type errors).

- [ ] **Step 5: Commit**

```bash
git add lib/env.ts .env.example
git commit -m "Add SEC_USER_AGENT env var for EDGAR fundamentals"
```

---

## Task 2: Types (`lib/types.ts`)

**Files:**
- Modify: `lib/types.ts`

- [ ] **Step 1: Append the SP3 types** (after `NewsArticle`):

```ts
// --- SP3: SEC EDGAR fundamentals ---

// A single XBRL fact as returned by SEC companyfacts/companyconcept.
export type SecFact = {
  start?: string;        // period start (flows); absent for instants
  end: string;           // period end (ISO yyyy-mm-dd)
  val: number;
  accn?: string;
  fy?: number;
  fp?: string;           // "FY" | "Q1".."Q4"
  form?: string;         // "10-K" | "10-Q" | ...
  filed?: string;        // filing date (ISO)
  frame?: string;
};

// Raw companyfacts payload (only the parts we read).
export type RawCompanyFacts = {
  cik?: number;
  entityName?: string;
  facts?: Record<string, Record<string, { label?: string; units?: Record<string, SecFact[]> }>>;
};

// Normalized reported values (stored as conceptsJson).
export type FundamentalConcepts = {
  revenue: number | null;
  netIncome: number | null;
  eps: number | null;               // diluted (fallback basic)
  operatingIncome: number | null;
  grossProfit: number | null;
  assets: number | null;
  liabilities: number | null;
  equity: number | null;
  currentAssets: number | null;
  currentLiabilities: number | null;
  sharesOutstanding: number | null;
};

// Provenance for honest "as of" labeling (stored as snapshot columns).
export type FundamentalsMeta = {
  fiscalYear: number | null;
  incomePeriodEnd: string | null;   // FY flow period end
  balanceSheetAsOf: string | null;  // latest balance-sheet instant
  filingForm: string | null;        // form providing the FY flows (e.g. "10-K")
  filedAt: string | null;           // filed date of those FY facts
};

export type ExtractedFundamentals = {
  concepts: FundamentalConcepts;
  meta: FundamentalsMeta;
};

// Display-ready metrics (null → "N/A"). Margins/ROE/ROA are fractions (×100 in UI).
export type FundamentalsView = {
  marketCap: number | null;
  peRatio: number | null;
  psRatio: number | null;
  grossMargin: number | null;
  roe: number | null;
  roa: number | null;
  operatingIncome: number | null;
  currentRatio: number | null;
  debtToEquity: number | null;
  assets: number | null;
  liabilities: number | null;
  equity: number | null;
  revenue: number | null;
  netIncome: number | null;
  eps: number | null;
};
```

- [ ] **Step 2: Verify types compile**

Run: `pnpm exec tsc --noEmit`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add lib/types.ts
git commit -m "Add SEC fundamentals domain types"
```

---

## Task 3: Schema + migration (`companies.cik` + `company_fundamentals`)

**Files:**
- Modify: `lib/db/schema.ts`
- Create: `drizzle/0002_*.sql` (generated)

- [ ] **Step 1: Add `cik` to the `companies` table** in `lib/db/schema.ts`. Add this line after `currency: text("currency"),`:

```ts
  cik: text("cik"), // SEC CIK, 10-digit zero-padded; resolved once, reused
```

- [ ] **Step 2: Add the `companyFundamentals` table** at the end of `lib/db/schema.ts` (all imports — `pgTable, uuid, text, date, timestamp, integer, jsonb, unique` — already exist at the top of the file):

```ts
export const companyFundamentals = pgTable("company_fundamentals", {
  id: uuid("id").primaryKey().defaultRandom(),
  companyId: uuid("company_id").notNull().references(() => companies.id),
  conceptsJson: jsonb("concepts_json").notNull(),
  fiscalYear: integer("fiscal_year"),
  incomePeriodEnd: date("income_period_end"),
  balanceSheetAsOf: date("balance_sheet_as_of"),
  filingForm: text("filing_form"),
  filedAt: date("filed_at"),
  source: text("source").notNull(),
  fetchedAt: timestamp("fetched_at", { withTimezone: true }).defaultNow().notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
}, (t) => [unique("uq_fundamentals_company").on(t.companyId)]);
```

- [ ] **Step 3: Generate the migration**

Run: `pnpm db:generate`
Expected: a new `drizzle/0002_*.sql` adds `companies.cik` and creates `company_fundamentals` with the unique constraint.

- [ ] **Step 4: Apply the migration to Neon**

Run: `pnpm db:migrate`
Expected: applies cleanly (no errors).

- [ ] **Step 5: Verify the table exists**

Run:
```bash
node -e "import('dotenv/config').then(async()=>{const {neon}=await import('@neondatabase/serverless');const sql=neon(process.env.DATABASE_URL);const r=await sql\`select column_name from information_schema.columns where table_name='company_fundamentals' order by 1\`;console.log(r.map(x=>x.column_name).join(','));const c=await sql\`select column_name from information_schema.columns where table_name='companies' and column_name='cik'\`;console.log('companies.cik:',c.length===1);})"
```
Expected: prints the `company_fundamentals` columns and `companies.cik: true`.

- [ ] **Step 6: Commit**

```bash
git add lib/db/schema.ts drizzle/
git commit -m "Add cik column and company_fundamentals snapshot table"
```

---

## Task 4: SEC provider (`lib/providers/sec.ts`)

**Files:**
- Create: `lib/providers/sec.ts`
- Test: `lib/providers/sec.test.ts`

- [ ] **Step 1: Write the failing test** for the pure `findCik` helper in `lib/providers/sec.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { findCik } from "./sec";

const MAP = {
  "0": { cik_str: 1045810, ticker: "NVDA", title: "NVIDIA CORP" },
  "1": { cik_str: 320193, ticker: "AAPL", title: "Apple Inc." },
};

describe("findCik", () => {
  it("returns the 10-digit zero-padded CIK for a known ticker", () => {
    expect(findCik(MAP, "AAPL")).toBe("0000320193");
    expect(findCik(MAP, "NVDA")).toBe("0001045810");
  });
  it("is case-insensitive", () => {
    expect(findCik(MAP, "aapl")).toBe("0000320193");
  });
  it("returns null when the ticker is absent", () => {
    expect(findCik(MAP, "ZZZZ")).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/providers/sec.test.ts`
Expected: FAIL — `findCik` is not exported / module not found.

- [ ] **Step 3: Implement `lib/providers/sec.ts`:**

```ts
import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { RawCompanyFacts } from "@/lib/types";

const TICKERS_URL = "https://www.sec.gov/files/company_tickers.json";
const FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts";
// Not hard-gated — SEC has no daily cap (only a ~10 req/s guideline that caching respects).
// The value only feeds the provider_state health row.
export const SEC_DAILY_LIMIT = 100000;

type TickerMap = Record<string, { cik_str: number; ticker: string; title: string }>;

export function findCik(map: TickerMap, ticker: string): string | null {
  const upper = ticker.toUpperCase();
  for (const k in map) {
    const entry = map[k];
    if (entry?.ticker?.toUpperCase() === upper) return String(entry.cik_str).padStart(10, "0");
  }
  return null;
}

export const sec = {
  name: "sec",
  async resolveCik(ticker: string): Promise<string | null> {
    if (!env.SEC_USER_AGENT) return null;
    try {
      const res = await fetch(TICKERS_URL, { headers: { "User-Agent": env.SEC_USER_AGENT } });
      if (!res.ok) throw new Error(`SEC tickers ${res.status}`);
      const cik = findCik((await res.json()) as TickerMap, ticker);
      await recordSuccess("sec", SEC_DAILY_LIMIT);
      return cik;
    } catch (e) {
      await recordError("sec", SEC_DAILY_LIMIT, String(e));
      return null;
    }
  },
  async fetchCompanyFacts(cik: string): Promise<RawCompanyFacts | null> {
    if (!env.SEC_USER_AGENT) return null;
    const url = `${FACTS_BASE}/CIK${cik}.json`;
    try {
      const res = await fetch(url, { headers: { "User-Agent": env.SEC_USER_AGENT } });
      if (!res.ok) throw new Error(`SEC facts ${res.status}`);
      const data = (await res.json()) as RawCompanyFacts;
      await recordSuccess("sec", SEC_DAILY_LIMIT);
      return data;
    } catch (e) {
      await recordError("sec", SEC_DAILY_LIMIT, String(e));
      return null;
    }
  },
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm test lib/providers/sec.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/providers/sec.ts lib/providers/sec.test.ts
git commit -m "Add SEC provider: CIK resolution and companyfacts fetch"
```

---

## Task 5: Pure extraction (`lib/fundamentals/extract.ts`)

**Files:**
- Create: `lib/fundamentals/extract.ts`
- Test: `lib/fundamentals/extract.test.ts`

- [ ] **Step 1: Write the failing test** in `lib/fundamentals/extract.test.ts`. The inline fixture exercises FY-vs-quarter selection, latest-instant selection, tag fallback, and missing tags:

```ts
import { describe, it, expect } from "vitest";
import { extractConcepts } from "./extract";
import type { RawCompanyFacts } from "@/lib/types";

function usd(facts: object[]) { return { units: { USD: facts } }; }

const RAW: RawCompanyFacts = {
  cik: 320193,
  entityName: "Apple Inc.",
  facts: {
    "us-gaap": {
      // No "Revenues" tag → must fall back to RevenueFromContractWithCustomerExcludingAssessedTax
      RevenueFromContractWithCustomerExcludingAssessedTax: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 391035000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
        { start: "2024-09-29", end: "2024-12-28", val: 124300000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }, // quarter — ignored for FY flow
        { start: "2022-09-25", end: "2023-09-30", val: 383285000000, fy: 2023, fp: "FY", form: "10-K", filed: "2023-11-03" }, // older FY
      ]),
      NetIncomeLoss: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 93736000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      EarningsPerShareDiluted: { units: { "USD/shares": [
        { start: "2023-10-01", end: "2024-09-28", val: 6.08, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ] } },
      OperatingIncomeLoss: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 123216000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      GrossProfit: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 180683000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      Assets: usd([
        { end: "2024-09-28", val: 364980000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
        { end: "2024-12-28", val: 344085000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }, // newer instant — wins
      ]),
      Liabilities: usd([{ end: "2024-12-28", val: 277327000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      StockholdersEquity: usd([{ end: "2024-12-28", val: 66758000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      AssetsCurrent: usd([{ end: "2024-12-28", val: 133240000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      LiabilitiesCurrent: usd([{ end: "2024-12-28", val: 144365000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
    },
    dei: {
      EntityCommonStockSharesOutstanding: { units: { shares: [
        { end: "2025-01-17", val: 15022073000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" },
      ] } },
    },
  },
};

describe("extractConcepts", () => {
  const { concepts, meta } = extractConcepts(RAW);

  it("picks the latest FY value for flow metrics (via tag fallback)", () => {
    expect(concepts.revenue).toBe(391035000000);   // 2024 FY, not the older FY or the quarter
    expect(concepts.netIncome).toBe(93736000000);
    expect(concepts.eps).toBe(6.08);
    expect(concepts.operatingIncome).toBe(123216000000);
    expect(concepts.grossProfit).toBe(180683000000);
  });

  it("picks the latest instant for balance-sheet metrics + shares", () => {
    expect(concepts.assets).toBe(344085000000);     // newer Q1 instant beats the FY instant
    expect(concepts.liabilities).toBe(277327000000);
    expect(concepts.equity).toBe(66758000000);
    expect(concepts.currentAssets).toBe(133240000000);
    expect(concepts.currentLiabilities).toBe(144365000000);
    expect(concepts.sharesOutstanding).toBe(15022073000);
  });

  it("records provenance from the income-statement anchor + the latest balance sheet", () => {
    expect(meta.fiscalYear).toBe(2024);
    expect(meta.incomePeriodEnd).toBe("2024-09-28");
    expect(meta.filingForm).toBe("10-K");
    expect(meta.filedAt).toBe("2024-11-01");
    expect(meta.balanceSheetAsOf).toBe("2024-12-28");
  });

  it("returns nulls for missing tags and an all-null result for empty input", () => {
    const partial = extractConcepts({ facts: { "us-gaap": {} } });
    expect(partial.concepts.revenue).toBeNull();
    expect(partial.meta.fiscalYear).toBeNull();
    const none = extractConcepts(null);
    expect(none.concepts.netIncome).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/fundamentals/extract.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lib/fundamentals/extract.ts`:**

```ts
import type { RawCompanyFacts, SecFact, ExtractedFundamentals } from "@/lib/types";

function durationDays(f: SecFact): number {
  if (!f.start) return 0;
  return (Date.parse(f.end) - Date.parse(f.start)) / 86400000;
}

// Most-recent full fiscal-year fact (fp:"FY", ~annual duration). Tie-break by latest filing.
function pickAnnualFlow(facts: SecFact[]): SecFact | null {
  const annual = facts.filter((f) => f.fp === "FY" && durationDays(f) >= 300);
  if (annual.length === 0) return null;
  return [...annual].sort(
    (a, b) => b.end.localeCompare(a.end) || (b.filed ?? "").localeCompare(a.filed ?? ""),
  )[0];
}

// Latest instant (balance-sheet / shares). Tie-break by latest filing.
function pickLatestInstant(facts: SecFact[]): SecFact | null {
  if (facts.length === 0) return null;
  return [...facts].sort(
    (a, b) => b.end.localeCompare(a.end) || (b.filed ?? "").localeCompare(a.filed ?? ""),
  )[0];
}

// First non-empty fact array among the candidate tags, in the given unit bucket.
function factsFor(raw: RawCompanyFacts, taxonomy: string, tags: string[], unit: string): SecFact[] {
  const tax = raw.facts?.[taxonomy];
  if (!tax) return [];
  for (const tag of tags) {
    const arr = tax[tag]?.units?.[unit];
    if (Array.isArray(arr) && arr.length > 0) return arr;
  }
  return [];
}

const EMPTY: ExtractedFundamentals = {
  concepts: {
    revenue: null, netIncome: null, eps: null, operatingIncome: null, grossProfit: null,
    assets: null, liabilities: null, equity: null, currentAssets: null, currentLiabilities: null,
    sharesOutstanding: null,
  },
  meta: { fiscalYear: null, incomePeriodEnd: null, balanceSheetAsOf: null, filingForm: null, filedAt: null },
};

export function extractConcepts(raw: RawCompanyFacts | null): ExtractedFundamentals {
  if (!raw || !raw.facts) return EMPTY;
  const gaap = (tags: string[], unit = "USD") => factsFor(raw, "us-gaap", tags, unit);

  const revenue = pickAnnualFlow(gaap(["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]));
  const netIncome = pickAnnualFlow(gaap(["NetIncomeLoss"]));
  const eps = pickAnnualFlow(gaap(["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares"));
  const operatingIncome = pickAnnualFlow(gaap(["OperatingIncomeLoss"]));
  const grossProfit = pickAnnualFlow(gaap(["GrossProfit"]));
  const assets = pickLatestInstant(gaap(["Assets"]));
  const liabilities = pickLatestInstant(gaap(["Liabilities"]));
  const equity = pickLatestInstant(gaap(["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]));
  const currentAssets = pickLatestInstant(gaap(["AssetsCurrent"]));
  const currentLiabilities = pickLatestInstant(gaap(["LiabilitiesCurrent"]));
  const shares = pickLatestInstant(factsFor(raw, "dei", ["EntityCommonStockSharesOutstanding"], "shares"));

  const anchor = netIncome ?? revenue;

  return {
    concepts: {
      revenue: revenue?.val ?? null,
      netIncome: netIncome?.val ?? null,
      eps: eps?.val ?? null,
      operatingIncome: operatingIncome?.val ?? null,
      grossProfit: grossProfit?.val ?? null,
      assets: assets?.val ?? null,
      liabilities: liabilities?.val ?? null,
      equity: equity?.val ?? null,
      currentAssets: currentAssets?.val ?? null,
      currentLiabilities: currentLiabilities?.val ?? null,
      sharesOutstanding: shares?.val ?? null,
    },
    meta: {
      fiscalYear: anchor?.fy ?? null,
      incomePeriodEnd: anchor?.end ?? null,
      balanceSheetAsOf: assets?.end ?? equity?.end ?? null,
      filingForm: anchor?.form ?? null,
      filedAt: anchor?.filed ?? null,
    },
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm test lib/fundamentals/extract.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/fundamentals/extract.ts lib/fundamentals/extract.test.ts
git commit -m "Add pure SEC concept extraction with period selection"
```

---

## Task 6: Pure derivation (`lib/fundamentals/derive.ts`)

**Files:**
- Create: `lib/fundamentals/derive.ts`
- Test: `lib/fundamentals/derive.test.ts`

- [ ] **Step 1: Write the failing test** in `lib/fundamentals/derive.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { deriveMetrics } from "./derive";
import type { FundamentalConcepts } from "@/lib/types";

const C: FundamentalConcepts = {
  revenue: 1000, netIncome: 100, eps: 5, operatingIncome: 200, grossProfit: 400,
  assets: 2000, liabilities: 1200, equity: 800, currentAssets: 600, currentLiabilities: 300,
  sharesOutstanding: 50,
};

describe("deriveMetrics", () => {
  it("computes price-derived multiples from the latest close", () => {
    const v = deriveMetrics(C, 10);
    expect(v.marketCap).toBe(500);       // 10 * 50
    expect(v.peRatio).toBe(2);           // 10 / 5
    expect(v.psRatio).toBe(0.5);         // 500 / 1000
  });
  it("computes price-independent ratios", () => {
    const v = deriveMetrics(C, 10);
    expect(v.grossMargin).toBe(0.4);     // 400 / 1000
    expect(v.roe).toBe(0.125);           // 100 / 800
    expect(v.roa).toBe(0.05);            // 100 / 2000
    expect(v.currentRatio).toBe(2);      // 600 / 300
    expect(v.debtToEquity).toBe(1.5);    // 1200 / 800
  });
  it("passes through raw financials", () => {
    const v = deriveMetrics(C, 10);
    expect(v.revenue).toBe(1000);
    expect(v.netIncome).toBe(100);
    expect(v.eps).toBe(5);
    expect(v.operatingIncome).toBe(200);
  });
  it("returns null for divide-by-zero / null inputs", () => {
    const v = deriveMetrics({ ...C, equity: 0, eps: null }, 10);
    expect(v.roe).toBeNull();            // /0
    expect(v.debtToEquity).toBeNull();   // /0
    expect(v.peRatio).toBeNull();        // eps null
    const noClose = deriveMetrics(C, null);
    expect(noClose.marketCap).toBeNull();
    expect(noClose.peRatio).toBeNull();
    expect(noClose.psRatio).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/fundamentals/derive.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lib/fundamentals/derive.ts`:**

```ts
import type { FundamentalConcepts, FundamentalsView } from "@/lib/types";

function div(a: number | null, b: number | null): number | null {
  if (a === null || b === null || b === 0) return null;
  return a / b;
}

export function deriveMetrics(c: FundamentalConcepts, latestClose: number | null): FundamentalsView {
  const marketCap = latestClose !== null && c.sharesOutstanding !== null
    ? latestClose * c.sharesOutstanding
    : null;
  return {
    marketCap,
    peRatio: div(latestClose, c.eps),
    psRatio: div(marketCap, c.revenue),
    grossMargin: div(c.grossProfit, c.revenue),
    roe: div(c.netIncome, c.equity),
    roa: div(c.netIncome, c.assets),
    operatingIncome: c.operatingIncome,
    currentRatio: div(c.currentAssets, c.currentLiabilities),
    debtToEquity: div(c.liabilities, c.equity),
    assets: c.assets,
    liabilities: c.liabilities,
    equity: c.equity,
    revenue: c.revenue,
    netIncome: c.netIncome,
    eps: c.eps,
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm test lib/fundamentals/derive.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/fundamentals/derive.ts lib/fundamentals/derive.test.ts
git commit -m "Add pure fundamentals ratio derivation"
```

---

## Task 7: Formatters (`lib/formatters.ts`)

**Files:**
- Modify: `lib/formatters.ts`
- Test: `lib/formatters.test.ts` (create if absent)

- [ ] **Step 1: Write the failing test** in `lib/formatters.test.ts` (append if the file exists):

```ts
import { describe, it, expect } from "vitest";
import { formatLargeCurrency, formatMultiple, formatRatioPercent } from "./formatters";

describe("formatLargeCurrency", () => {
  it("scales to T/B/M and shows N/A for null", () => {
    expect(formatLargeCurrency(3_420_000_000_000)).toBe("$3.42T");
    expect(formatLargeCurrency(1_230_000_000)).toBe("$1.23B");
    expect(formatLargeCurrency(456_700_000)).toBe("$456.70M");
    expect(formatLargeCurrency(-1_230_000_000)).toBe("-$1.23B");
    expect(formatLargeCurrency(null)).toBe("N/A");
  });
});
describe("formatMultiple", () => {
  it("appends × and shows N/A for null", () => {
    expect(formatMultiple(28.41)).toBe("28.41×");
    expect(formatMultiple(null)).toBe("N/A");
  });
});
describe("formatRatioPercent", () => {
  it("renders a fraction as a percent and N/A for null", () => {
    expect(formatRatioPercent(0.243)).toBe("24.3%");
    expect(formatRatioPercent(null)).toBe("N/A");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/formatters.test.ts`
Expected: FAIL — functions not exported.

- [ ] **Step 3: Append to `lib/formatters.ts`:**

```ts
export function formatLargeCurrency(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  return `${sign}$${abs.toLocaleString("en-US")}`;
}
export function formatMultiple(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  return `${v.toFixed(2)}×`;
}
export function formatRatioPercent(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  return `${(v * 100).toFixed(1)}%`;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm test lib/formatters.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/formatters.ts lib/formatters.test.ts
git commit -m "Add fundamentals display formatters"
```

---

## Task 8: Snapshot repo + `setCik` (`lib/db/fundamentals.ts`, `lib/db/companies.ts`)

**Files:**
- Create: `lib/db/fundamentals.ts`
- Modify: `lib/db/companies.ts`

No unit test (thin DB wrappers; covered by the Task 9 integration test). Verify via `tsc`.

- [ ] **Step 1: Add `setCik` to `lib/db/companies.ts`** (the file already imports `eq` from `drizzle-orm` and `companies` from `./schema`):

```ts
export async function setCik(companyId: string, cik: string): Promise<void> {
  await db.update(companies).set({ cik, updatedAt: new Date() }).where(eq(companies.id, companyId));
}
```

- [ ] **Step 2: Create `lib/db/fundamentals.ts`:**

```ts
import { db } from "./client";
import { companyFundamentals } from "./schema";
import { eq } from "drizzle-orm";
import type { FundamentalConcepts, FundamentalsMeta } from "@/lib/types";

export type FundamentalsRow = typeof companyFundamentals.$inferSelect;

export async function getFundamentalsSnapshot(companyId: string): Promise<FundamentalsRow | undefined> {
  const [row] = await db.select().from(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
  return row;
}

export async function upsertFundamentalsSnapshot(input: {
  companyId: string;
  conceptsJson: FundamentalConcepts;
  meta: FundamentalsMeta;
  source: string;
  expiresAt: Date;
}): Promise<void> {
  const values = {
    companyId: input.companyId,
    conceptsJson: input.conceptsJson,
    fiscalYear: input.meta.fiscalYear,
    incomePeriodEnd: input.meta.incomePeriodEnd,
    balanceSheetAsOf: input.meta.balanceSheetAsOf,
    filingForm: input.meta.filingForm,
    filedAt: input.meta.filedAt,
    source: input.source,
    fetchedAt: new Date(),
    expiresAt: input.expiresAt,
  };
  await db.insert(companyFundamentals).values(values).onConflictDoUpdate({
    target: companyFundamentals.companyId,
    set: { ...values, fetchedAt: new Date() },
  });
}
```

- [ ] **Step 3: Verify it compiles**

Run: `pnpm exec tsc --noEmit`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add lib/db/fundamentals.ts lib/db/companies.ts
git commit -m "Add fundamentals snapshot repo and setCik"
```

---

## Task 9: Fundamentals service (`lib/services/fundamentals-service.ts`)

**Files:**
- Create: `lib/services/fundamentals-service.ts`
- Test: `lib/services/fundamentals-service.test.ts` (integration — live Neon, SEC mocked)

- [ ] **Step 1: Write the failing integration test** in `lib/services/fundamentals-service.test.ts`. It mocks the SEC provider and `price-service` (so no live HTTP) but uses the real Neon DB (per `dotenv/config`). It seeds a company directly:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the SEC provider — no live HTTP.
vi.mock("@/lib/providers/sec", () => ({
  SEC_DAILY_LIMIT: 100000,
  findCik: () => null,
  sec: {
    name: "sec",
    resolveCik: vi.fn(async () => "0000000001"),
    fetchCompanyFacts: vi.fn(async () => ({
      facts: {
        "us-gaap": {
          Revenues: { units: { USD: [
            { start: "2023-01-01", end: "2023-12-31", val: 1000, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
          ] } },
          NetIncomeLoss: { units: { USD: [
            { start: "2023-01-01", end: "2023-12-31", val: 100, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
          ] } },
          Assets: { units: { USD: [{ end: "2023-12-31", val: 2000, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" }] } },
          StockholdersEquity: { units: { USD: [{ end: "2023-12-31", val: 800, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" }] } },
        },
        dei: { EntityCommonStockSharesOutstanding: { units: { shares: [
          { end: "2023-12-31", val: 50, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
        ] } } },
      },
    })),
  },
}));

// Mock price-service so the company is ensured with a known close + asset type.
vi.mock("@/lib/services/price-service", () => ({
  getTickerData: vi.fn(async () => ({ bars: [{ date: "2024-01-02", open: 0, high: 0, low: 0, close: 10, adjClose: null, volume: 0 }] })),
}));

import { getFundamentals } from "./fundamentals-service";
import { db } from "@/lib/db/client";
import { companies, companyFundamentals } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { sec } from "@/lib/providers/sec";

async function reseed(ticker: string, assetType: string) {
  const [c] = await db.select().from(companies).where(eq(companies.ticker, ticker));
  if (c) {
    await db.delete(companyFundamentals).where(eq(companyFundamentals.companyId, c.id));
    await db.delete(companies).where(eq(companies.id, c.id));
  }
  const [row] = await db.insert(companies)
    .values({ ticker, name: ticker, assetType, currency: "USD" }).returning();
  return row;
}

describe("fundamentals-service (live Neon, SEC mocked)", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns not_applicable for ETFs/indexes", async () => {
    await reseed("TST_ETF", "etf");
    const r = await getFundamentals("TST_ETF");
    expect(r.status).toBe("not_applicable");
    expect(sec.fetchCompanyFacts).not.toHaveBeenCalled();
  });

  it("resolves CIK, fetches, extracts, derives, and caches (cache-first on 2nd call)", async () => {
    await reseed("TST_STK", "stock");
    const r1 = await getFundamentals("TST_STK");
    expect(r1.status).toBe("ok");
    expect(r1.view?.revenue).toBe(1000);
    expect(r1.view?.marketCap).toBe(500);      // close 10 * shares 50
    expect(r1.view?.peRatio).toBeNull();        // no EPS in fixture
    expect(r1.asOf?.fiscalYear).toBe(2023);
    expect(r1.asOf?.edgarUrl).toContain("CIK=0000000001");
    expect(sec.fetchCompanyFacts).toHaveBeenCalledTimes(1);

    const r2 = await getFundamentals("TST_STK");
    expect(r2.status).toBe("ok");
    expect(sec.fetchCompanyFacts).toHaveBeenCalledTimes(1); // served from cache — no refetch
  });

  it("returns unavailable when no CIK can be resolved and there is no cache", async () => {
    await reseed("TST_NOCIK", "stock");
    (sec.resolveCik as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    const r = await getFundamentals("TST_NOCIK");
    expect(r.status).toBe("unavailable");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/services/fundamentals-service.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `lib/services/fundamentals-service.ts`:**

```ts
import { getCompanyByTicker, setCik } from "@/lib/db/companies";
import { getTickerData } from "@/lib/services/price-service";
import { getFundamentalsSnapshot, upsertFundamentalsSnapshot, type FundamentalsRow } from "@/lib/db/fundamentals";
import { sec } from "@/lib/providers/sec";
import { extractConcepts } from "@/lib/fundamentals/extract";
import { deriveMetrics } from "@/lib/fundamentals/derive";
import type { FundamentalConcepts, FundamentalsMeta, FundamentalsView } from "@/lib/types";

export type FundamentalsStatus = "ok" | "not_applicable" | "unavailable" | "error";
export type FundamentalsAsOf = FundamentalsMeta & { edgarUrl: string | null };
export type FundamentalsResult = {
  status: FundamentalsStatus;
  view: FundamentalsView | null;
  asOf: FundamentalsAsOf | null;
  source: "sec_edgar" | null;
};

const TTL_MS = 7 * 24 * 60 * 60 * 1000;
const edgarUrl = (cik: string) =>
  `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&type=10-K`;

const metaFromSnap = (s: FundamentalsRow): FundamentalsMeta => ({
  fiscalYear: s.fiscalYear,
  incomePeriodEnd: s.incomePeriodEnd,
  balanceSheetAsOf: s.balanceSheetAsOf,
  filingForm: s.filingForm,
  filedAt: s.filedAt,
});

function ok(concepts: FundamentalConcepts, meta: FundamentalsMeta, cik: string | null, latestClose: number | null): FundamentalsResult {
  return {
    status: "ok",
    view: deriveMetrics(concepts, latestClose),
    asOf: { ...meta, edgarUrl: cik ? edgarUrl(cik) : null },
    source: "sec_edgar",
  };
}

export async function getFundamentals(ticker: string): Promise<FundamentalsResult> {
  // Ensures the company exists + provides the latest cached close (no extra provider call when bars are fresh).
  const priceData = await getTickerData(ticker, "1y");
  const company = await getCompanyByTicker(ticker);
  if (!company) return { status: "unavailable", view: null, asOf: null, source: null };
  if (company.assetType !== "stock") return { status: "not_applicable", view: null, asOf: null, source: null };

  const latestClose = priceData.bars.at(-1)?.close ?? null;
  const snap = await getFundamentalsSnapshot(company.id);

  // Cache-first.
  if (snap && snap.expiresAt.getTime() > Date.now()) {
    return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), company.cik, latestClose);
  }

  // Ensure CIK.
  let cik = company.cik;
  if (!cik) {
    cik = await sec.resolveCik(company.ticker);
    if (cik) await setCik(company.id, cik);
  }
  if (!cik) {
    if (snap) return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), null, latestClose);
    return { status: "unavailable", view: null, asOf: null, source: null };
  }

  // Fetch + extract + cache.
  const raw = await sec.fetchCompanyFacts(cik);
  if (!raw) {
    if (snap) return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), cik, latestClose);
    return { status: "error", view: null, asOf: null, source: null };
  }
  const { concepts, meta } = extractConcepts(raw);
  await upsertFundamentalsSnapshot({
    companyId: company.id, conceptsJson: concepts, meta, source: "sec_edgar",
    expiresAt: new Date(Date.now() + TTL_MS),
  });
  return ok(concepts, meta, cik, latestClose);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm test lib/services/fundamentals-service.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/services/fundamentals-service.ts lib/services/fundamentals-service.test.ts
git commit -m "Add fundamentals service: cache-first SEC orchestration"
```

---

## Task 10: API route (`app/api/fundamentals/[symbol]/route.ts`)

**Files:**
- Create: `app/api/fundamentals/[symbol]/route.ts`

- [ ] **Step 1: Implement the route** (mirrors `app/api/memo/[symbol]/route.ts`):

```ts
import { type NextRequest } from "next/server";
import { z } from "zod";
import { getFundamentals } from "@/lib/services/fundamentals-service";

export const dynamic = "force-dynamic";
export const maxDuration = 30; // SEC companyfacts fetch can be a few MB

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);

export async function GET(_request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  try {
    return Response.json(await getFundamentals(t.data.toUpperCase()));
  } catch (e) {
    // Surface as a card-renderable error rather than an HTTP failure.
    return Response.json({ status: "error", view: null, asOf: null, source: null, detail: String(e) });
  }
}
```

- [ ] **Step 2: Verify it compiles + the dev server serves it**

Run: `pnpm exec tsc --noEmit`
Expected: PASS.

Then (manually, optional here — fully exercised in Task 12): `curl -s localhost:3000/api/fundamentals/AAPL` returns JSON with `status`.

- [ ] **Step 3: Commit**

```bash
git add app/api/fundamentals/
git commit -m "Add /api/fundamentals/[symbol] route"
```

---

## Task 11: Fundamentals card + wire into ticker page

**Files:**
- Create: `components/fundamentals-card.tsx`
- Modify: `app/ticker/[symbol]/page.tsx`

- [ ] **Step 1: Implement `components/fundamentals-card.tsx`** (client; mirrors `memo-card.tsx`):

```tsx
"use client";
import { useCallback, useEffect, useState } from "react";
import type { FundamentalsResult } from "@/lib/services/fundamentals-service";
import { formatLargeCurrency, formatMultiple, formatRatioPercent } from "@/lib/formatters";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-0.5 text-sm">
      <span className="text-neutral-500">{label}</span>
      <span className="font-mono text-neutral-100">{value}</span>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <h3 className="text-xs font-medium uppercase text-neutral-500">{title}</h3>
      <div className="mt-1">{children}</div>
    </div>
  );
}

export function FundamentalsCard({ symbol }: { symbol: string }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<FundamentalsResult | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/fundamentals/${symbol}`);
      setData(await res.json());
    } catch {
      setData({ status: "error", view: null, asOf: null, source: null });
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => { void load(); }, [load]);

  if (loading) return <p className="text-sm text-neutral-500">Loading fundamentals…</p>;
  if (!data || data.status === "error")
    return (
      <div className="text-sm text-neutral-500">
        Couldn&apos;t load fundamentals. <button onClick={() => load()} className="underline">Try again</button>
      </div>
    );
  if (data.status === "not_applicable")
    return <p className="text-sm text-neutral-500">Fundamentals aren&apos;t available for ETFs/indexes.</p>;
  if (data.status === "unavailable" || !data.view)
    return <p className="text-sm text-neutral-500">Fundamentals unavailable — SEC data couldn&apos;t be fetched.</p>;

  const v = data.view;
  const a = data.asOf;
  return (
    <div className="rounded-lg ring-1 ring-neutral-800 p-4">
      <div className="grid grid-cols-1 gap-x-8 sm:grid-cols-2">
        <Group title="Valuation">
          <Row label="Market Cap" value={formatLargeCurrency(v.marketCap)} />
          <Row label="P/E (FY EPS)" value={formatMultiple(v.peRatio)} />
          <Row label="P/S" value={formatMultiple(v.psRatio)} />
        </Group>
        <Group title="Profitability">
          <Row label="Gross Margin" value={formatRatioPercent(v.grossMargin)} />
          <Row label="ROE" value={formatRatioPercent(v.roe)} />
          <Row label="ROA" value={formatRatioPercent(v.roa)} />
          <Row label="Operating Income" value={formatLargeCurrency(v.operatingIncome)} />
        </Group>
        <Group title="Financial Health">
          <Row label="Current Ratio" value={formatMultiple(v.currentRatio)} />
          <Row label="Debt/Equity" value={formatMultiple(v.debtToEquity)} />
          <Row label="Assets" value={formatLargeCurrency(v.assets)} />
          <Row label="Liabilities" value={formatLargeCurrency(v.liabilities)} />
          <Row label="Equity" value={formatLargeCurrency(v.equity)} />
        </Group>
        <Group title="Latest Financials">
          <Row label="Revenue" value={formatLargeCurrency(v.revenue)} />
          <Row label="Net Income" value={formatLargeCurrency(v.netIncome)} />
          <Row label="EPS (diluted)" value={v.eps === null ? "N/A" : `$${v.eps.toFixed(2)}`} />
        </Group>
      </div>
      <div className="mt-4 border-t border-neutral-800 pt-2 text-xs text-neutral-600">
        Source: SEC EDGAR
        {a?.fiscalYear ? ` · FY${a.fiscalYear} ${a.filingForm ?? ""}${a.filedAt ? ` filed ${a.filedAt}` : ""}` : ""}
        {a?.balanceSheetAsOf ? ` · balance sheet as of ${a.balanceSheetAsOf}` : ""}
        {a?.edgarUrl ? <> · <a href={a.edgarUrl} target="_blank" rel="noopener noreferrer" className="underline">filings on SEC.gov</a></> : null}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into `app/ticker/[symbol]/page.tsx`.** Add the import near the other component imports:

```ts
import { FundamentalsCard } from "@/components/fundamentals-card";
```

Then insert a new section between the "Daily memo" section and the "Recent news" section:

```tsx
      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Fundamentals</h2>
        <p className="mb-2 text-xs text-neutral-600">From official SEC filings.</p>
        <FundamentalsCard symbol={data.ticker} />
      </section>
```

- [ ] **Step 3: Verify it compiles**

Run: `pnpm exec tsc --noEmit`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add components/fundamentals-card.tsx app/ticker/
git commit -m "Add fundamentals card and wire into ticker page"
```

---

## Task 12: E2E smoke + full verification

**Files:**
- Modify: `tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Extend `tests/e2e/smoke.spec.ts`** to intercept `/api/fundamentals` with a fixture and assert the card renders. Add a route handler alongside the existing `/api/memo` interception (place the fulfill before navigation):

```ts
  await page.route("**/api/fundamentals/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        source: "sec_edgar",
        asOf: { fiscalYear: 2024, incomePeriodEnd: "2024-09-28", balanceSheetAsOf: "2024-12-28", filingForm: "10-K", filedAt: "2024-11-01", edgarUrl: "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K" },
        view: {
          marketCap: 3420000000000, peRatio: 28.41, psRatio: 8.7, grossMargin: 0.462, roe: 1.5, roa: 0.28,
          operatingIncome: 123216000000, currentRatio: 0.92, debtToEquity: 4.15,
          assets: 364980000000, liabilities: 308030000000, equity: 56950000000,
          revenue: 391035000000, netIncome: 93736000000, eps: 6.08,
        },
      }),
    }),
  );
```

Then add assertions after the existing memo/news assertions:

```ts
  await expect(page.getByRole("heading", { name: "Fundamentals" })).toBeVisible();
  await expect(page.getByText("Market Cap")).toBeVisible();
  await expect(page.getByText("$3.42T")).toBeVisible();
```

- [ ] **Step 2: Run the E2E smoke**

Run: `pnpm test:e2e`
Expected: PASS (the fundamentals card renders from the fixture; news + memo still pass).

- [ ] **Step 3: Run the full unit/integration suite + typecheck + build**

Run: `pnpm test && pnpm exec tsc --noEmit && pnpm build`
Expected: all unit + integration tests pass; tsc clean; production build green. (Do NOT run `pnpm lint` — OOM locally; CI covers it.)

- [ ] **Step 4: LIVE SEC smoke (required — mocks can't catch real tagging quirks).** Start the dev server and hit the real endpoint:

```bash
pnpm dev &   # or use an already-running server on :3000
sleep 4
curl -s "localhost:3000/api/fundamentals/AAPL" | python3 -m json.tool | head -40
curl -s "localhost:3000/api/fundamentals/NVDA" | python3 -m json.tool | head -40
curl -s "localhost:3000/api/fundamentals/SPY"  | python3 -c "import sys,json;print('SPY status:',json.load(sys.stdin)['status'])"
```
Expected: AAPL + NVDA return `status:"ok"` with sensible non-null `view.revenue`, `view.netIncome`, `view.marketCap`, and an `asOf.fiscalYear`; SPY returns `status:"not_applicable"`. If a core figure is unexpectedly null, inspect the real concept tags for that company and extend the fallback lists in `extract.ts` (then re-run unit tests).

- [ ] **Step 5: Live visual check.** Load `http://localhost:3000/ticker/AAPL` in a browser (or a throwaway Playwright screenshot spec) and confirm the fundamentals card renders the four groups with real numbers and the SEC EDGAR footer link. (The preview MCP was unreachable in this environment last time; a one-off Playwright spec that screenshots the live page is the reliable fallback — delete it after.)

- [ ] **Step 6: Final commit (if Step 4 required an `extract.ts` fallback tweak)**

```bash
git add -A
git commit -m "Verify live SEC fundamentals for AAPL/NVDA"
```

---

## Self-Review (completed during planning)

**Spec coverage (§8):**
- §8.1 scope (stocks only, one companyfacts call, client-fetched, latest-annual) → Tasks 4, 9, 10, 11. ✓
- §8.2 data model (`cik`, `company_fundamentals`, `sec` provider_state row, `SEC_USER_AGENT`) → Tasks 1, 3; the `sec` provider_state row is created lazily by `ensureRow` on first `recordSuccess`/`recordError` in Task 4 (no migration needed — matches how `finnhub`/`gemini` rows are created). ✓
- §8.3 provider (`resolveCik`, `fetchCompanyFacts`, UA, degrade) → Task 4. ✓
- §8.4 pure extract/derive (tag fallback, unit buckets, period selection, guarded ratios) → Tasks 5, 6. ✓
- §8.5 service + route (cache-first, CIK ensure, stale fallback, `not_applicable`/`unavailable`/`error`, `maxDuration`) → Tasks 9, 10. ✓
- §8.6 UI (four groups, N/A, trust footer + EDGAR link, placed after memo) → Task 11. ✓
- §8.7 error handling (degrade paths) → Tasks 4, 9, 11. ✓
- §8.8 testing (unit extract/derive/findCik, integration cache-first/not_applicable/unavailable, E2E intercept, live smoke) → Tasks 4, 5, 6, 9, 12. ✓
- §8.9 acceptance criteria → verified in Task 12. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. ✓

**Type consistency:** `FundamentalConcepts`/`FundamentalsMeta`/`ExtractedFundamentals`/`FundamentalsView`/`SecFact`/`RawCompanyFacts` defined in Task 2 and used identically in Tasks 4–11. `extractConcepts(raw): ExtractedFundamentals` (Task 5) ↔ consumed as `{ concepts, meta }` in Task 9. `deriveMetrics(concepts, latestClose)` (Task 6) ↔ called in Task 9 + the service's `ok()`. `getFundamentals(ticker): FundamentalsResult` (Task 9) ↔ imported by the route (Task 10) + the card's `FundamentalsResult` type (Task 11). `findCik`/`sec` (Task 4) ↔ mocked in Task 9's test with matching shapes. ✓

**Note on a DRY decision:** `conceptsJson` stores only the 11 reported values; the 5 provenance fields (`fiscalYear`/`incomePeriodEnd`/`balanceSheetAsOf`/`filingForm`/`filedAt`) live only in columns — no duplication, and the spec's listed columns are honored.
