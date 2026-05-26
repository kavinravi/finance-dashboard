# Finance Dashboard Rebuild — Design Spec

**Date:** 2026-05-25
**Branch:** `rebuild-nextjs-mvp`
**Status:** Approved design; ready for implementation planning.

This spec is the de-risked, authoritative architecture for rebuilding `finance-dashboard`. It supersedes the architecture in `plan.md` where they differ (notably: single app instead of a two-service split). `plan.md` remains the broader product-vision reference (page-level features, prompt rules, compliance posture, risk register).

---

## 1. Goal & posture

A deployed, personal investing-**research** dashboard: search a ticker or company name, open a research page with prices, technical indicators, a one-target comparison, recent news, source-linked sentiment, a Gemini-generated daily memo, and basic fundamentals.

- **Research assistant, not investment advice.** Never emits buy/sell/hold.
- **Trust through evidence.** Every memo claim cites source article IDs; sentiment is labeled "News Tone" with article count/confidence; data freshness is shown honestly. (A primary user distrusts AI summaries, so source-linking and transparency are core requirements, not polish.)
- **Personal / non-commercial use.** This is a hard constraint, not a preference — see §3.

## 2. What changed from `plan.md` and why

`plan.md` assumed a Next.js frontend + a separate Python FastAPI worker, with local FinBERT for sentiment, FMP for news + fundamentals, and Alpha Vantage as a news/sentiment source. Pre-build verification (2026-05-25) invalidated those assumptions:

| `plan.md` assumption | Verified reality (2026) | Decision |
|---|---|---|
| FMP free covers search, profile, prices, **news, fundamentals** | FMP free ("Basic", 250 calls/day + 500MB/30-day bandwidth) covers **search + profile + EOD prices only**; **news & fundamentals are paid** (Starter $22/mo+). Use `/stable/` routes (legacy `/api/v3/` is deprecated). | FMP = search/profile/prices only. News and fundamentals sourced elsewhere. |
| Alpha Vantage for news/sentiment | Free tier is now **25 requests/day**, 5/min — unusable for recurring ingestion. | **Dropped entirely.** No AV key needed. |
| Local FinBERT for sentiment | Needs **~2GB RAM**; won't fit free/cheap (512MB–1GB) containers; no reliable free hosted endpoint. FinBERT was the **only** reason for a separate Python service. | **Dropped from MVP.** Sentiment comes from Gemini (§4). FinBERT becomes an optional later "Labs" add-on if a paid host is ever justified. |
| `gemini-3-flash-preview` | Exists (preview); stable `gemini-3.5-flash` also available. Preview models can be deprecated. | Default to stable `gemini-3.5-flash`; keep preview configurable via env. |

**Net effect:** no Python, no second service, no PyTorch. The product is **one Next.js app on Vercel.** This removes two-service deployment, CORS, shared-OpenAPI-types coordination, and ~$5–10/mo of always-on ML hosting.

## 3. Data sources & the personal-use constraint

| Need | Source | Tier | Notes |
|---|---|---|---|
| Search, company profile, EOD daily prices | **FMP** (`/stable/`) | Free, 250/day | Binding limit is 250 **calls/day** + 500MB/30-day bandwidth. Cache hard. |
| Price + search fallback | **Yahoo** (`yahoo-finance2` + RSS) | Free, keyless | Scrape-based; periodic 401/crumb/429 breakage. Best-effort fallback only. |
| Company news | **Finnhub** `/company-news` | Free, 60/min, no daily cap | **Personal/non-commercial use only.** |
| News fallback | **Yahoo Finance RSS** (+ Google News RSS) | Free, keyless | Verified working 2026-05-25. Personal-use only. |
| Fundamentals & filings | **SEC EDGAR** (CompanyFacts/submissions JSON) | Free, official | Primary (FMP fundamentals are paid). Requires a contact `User-Agent`. |
| Sentiment (News Tone) + daily memo | **Gemini** (`gemini-3.5-flash` default) | Free tier (limits live in AI Studio) | Structured JSON output. |

**Hard constraint:** Finnhub-free, Yahoo, and Google News are all licensed for **personal/non-commercial use only**. The app therefore stays **private (password-gated)** and must not publicly redistribute provider data. Making it public/commercial later would require paid data tiers (FMP Starter+, a commercial news API, etc.). This aligns with `plan.md`'s "no public redistribution" non-goal.

**Deferred paid option:** Nasdaq Data Link "Sharadar Core US Fundamentals" (SF1, ~$50/mo individual, unconfirmed) — a clean normalized-fundamentals upgrade path if SEC XBRL parsing (SP3) proves too messy. Not used in the MVP.

## 4. Architecture (cross-cutting, all phases)

- **One Next.js app** (App Router + TypeScript) at the **repo root**, deployed on Vercel.
- **All provider calls are server-side** (route handlers / server components) so API keys never reach the client.
- **Sentiment + memo via Gemini:** the same batched, structured-JSON Gemini calls produce per-article News Tone *and* the source-linked daily memo. Tradeoff accepted: LLM-derived tone is less deterministic than a dedicated finance model and carries small per-call cost, in exchange for zero ML infra.
- **Indicators in TypeScript:** MA/RSI/MACD/volatility/returns are pure functions (math ported from `legacy/app.py`), computed on the fly from cached price bars (no precomputed indicator table).
- **Storage = bounded cache,** not a warehouse: **Neon Postgres + Drizzle ORM.** Same Postgres dialect locally (Docker or a Neon branch) and in prod — no SQLite/Postgres dialect drift. Freshness/retention rules per `plan.md` §2.
- **No scheduler in the MVP:** fetch/compute on page load when the cache is stale, then persist. (A daily refresh job can be added later if "what changed since yesterday" trends are wanted.)
- **Auth:** simple password gate (also enforces the private/personal-use posture).
- **Frontend libs:** Tailwind + shadcn/ui, Recharts + lightweight-charts, TanStack Query/Table, Zod for boundary validation.

### Known risks / watch-items (carried into implementation)
- **Vercel function duration:** on-demand memo generation (news fetch + Gemini) can be slow (SP2). Mitigate: cache-first, possibly generate-then-poll, set `maxDuration`.
- **FMP 250/day budget:** enforce via `provider_state` call counting + aggressive caching.
- **Yahoo fragility:** treat as fallback only; never the sole source for a page.

## 5. Phasing — four sub-projects

Each sub-project gets its own spec → plan → build cycle so we never build on an unproven foundation.

1. **SP1 — Foundation + Search + Prices + Comparison** *(detailed below; build first)*. Usable slice: search "NVIDIA" → NVDA research page with price chart + AMD comparison.
2. **SP2 — News + Sentiment + Memo:** Finnhub + Yahoo RSS ingestion, dedupe, Gemini News Tone + source-linked daily memo, news table + tone meter + memo UI. *(detailed in §7 below)*
3. **SP3 — Fundamentals:** SEC EDGAR CIK mapping + CompanyFacts (minimal concepts), fundamentals card. Adds `cik` to `companies`. *(detailed in §8 below)*
4. **SP4 — Deploy & harden:** Vercel + Neon prod, password gate, provider-health page (reads `provider_state`), cache retention/pruning.

---

## 6. SP1 detailed design

### 6.1 Structure & stack

Single app at repo root:

```
finance-dashboard/
  app/                       # Next.js App Router
    page.tsx                 # search-first homepage
    ticker/[symbol]/page.tsx # company research page
    compare/page.tsx
    api/                     # server-side route handlers
      search/route.ts
      prices/[symbol]/route.ts
      compare/route.ts
  components/                # SearchBar, PriceChart, ReturnsTable, ComparisonChart, RecentSearches, ...
  lib/
    providers/               # base.ts, fmp.ts, yahoo.ts
    indicators/              # returns.ts, moving-averages.ts, rsi.ts, macd.ts, volatility.ts
    db/                      # drizzle schema + client
    services/                # search-service, price-service, comparison-service
    types.ts, formatters.ts
  drizzle/                   # migrations
  legacy/                    # archived Streamlit app (done)
  .env.example
  package.json, next.config.ts, tsconfig.json, tailwind.config.ts
```

**Stack:** Next.js App Router, TypeScript, Tailwind, shadcn/ui, Drizzle + Neon Postgres, Zod, TanStack Query/Table, Recharts + lightweight-charts.

**Env (finalized at scaffold):** `FMP_API_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash`, `GEMINI_PREVIEW_MODEL=gemini-3-flash-preview` (optional), `SEC_USER_AGENT`, `DATABASE_URL` (Neon), `APP_PASSWORD` (SP4), `FMP_DAILY_LIMIT=250`. A committed `.env.example` documents all of these.

### 6.2 Data model (Drizzle / Postgres) — four tables

- **`companies`** — resolved securities, including ETFs/benchmarks (SPY/QQQ): `id`, `ticker` (unique, uppercase), `name`, `asset_type` (`stock`/`etf`/`index`), `exchange`, `sector`, `industry`, `currency`, `last_profile_refresh_at`, `created_at`, `updated_at`. (`cik` added in SP3.)
- **`price_bars_daily`** — `id`, `company_id` FK, `date`, `open`, `high`, `low`, `close`, `adj_close` (nullable), `volume`, `source`. **Unique `(company_id, date, source)`.**
- **`recent_searches`** — `id`, `query`, `resolved_ticker` (nullable), `created_at`.
- **`provider_state`** — `provider`, `calls_today`, `daily_limit`, `reset_at`, `last_success_at`, `last_error_at`, `last_error`. (DB-backed because serverless has no persistent memory to count the FMP budget; also feeds the SP4 health page.)

(`technical_indicators_daily` from `plan.md` §9 is intentionally omitted — indicators are computed on the fly. Saved watchlist is deferred to SP4; SP1 ships only `recent_searches`.)

### 6.3 Provider layer (`lib/providers/`, server-side)

- **`base.ts`** — shared types (`PriceBar`, `CompanyProfile`, `SearchResult`) and a thin `Provider` contract.
- **`fmp.ts`** — `search(query)`, `profile(ticker)`, `dailyPrices(ticker, from, to)` against FMP `/stable/` routes. Every call updates `provider_state` (increment `calls_today`, record success/error). Respects `FMP_DAILY_LIMIT`.
- **`yahoo.ts`** — `yahoo-finance2` for price + search **fallback**, wrapped in try/catch (scrape-fragile); failures recorded to `provider_state` and degraded silently.

### 6.4 Services (`lib/services/`)

- **`search-service`** — normalize query → exact ticker hit in `companies` → else FMP `search()` → log `recent_searches` → return candidates (ticker, name, exchange, asset type, source).
- **`price-service`** — `getPrices(ticker, range)`: check freshness (latest cached bar vs last trading day); if stale **and** FMP budget available → FMP `dailyPrices()`, else Yahoo, else serve cache; upsert bars; return series. Computes returns + indicators via `lib/indicators`.
- **`comparison-service`** — `getComparison(primary, comparison, range)`: fetch both series, align on common dates, compute relative return, volatility, drawdown, and a performance table.

### 6.5 Indicators (`lib/indicators/`)

Pure functions over a bar series, unit-tested test-first against golden values (cross-checked vs `legacy/app.py`, using `legacy/data/SPY.csv` as a fixture):
- **Returns:** 1D, 5D, 1M, 3M, 6M, YTD, 1Y (nearest bar at/just-before the lookback date → % change to latest).
- **Moving averages:** MA10, MA20, MA50.
- **RSI14**, **MACD(12/26/9)**, **5-day volatility** (rolling std of daily returns).

### 6.6 Data flows

1. **Search:** homepage `SearchBar` → `GET /api/search?q=` → `search-service` → candidates → user selects → navigate to `/ticker/[symbol]`.
2. **Ticker page** (server component): load profile (cache → FMP) + bars (cache → FMP → Yahoo) → compute returns + indicators → render header (price, name, period returns) + `PriceChart` (MA overlays, range selector 1M/3M/6M/YTD/1Y) + comparison selector.
3. **Compare:** `/compare?primary=&comparison=` → `comparison-service` → `ComparisonChart` overlay + relative return + volatility + drawdown + performance table.

### 6.7 Error handling — "degrade, never crash"

- **Fallback chain:** FMP → Yahoo → stale cache. A provider failure never breaks a page.
- **Staleness honesty:** when serving stale cache (provider down or budget spent), show a "data as of \<date\>" badge — never fail silently or misrepresent freshness.
- **FMP budget exhaustion:** when `calls_today >= FMP_DAILY_LIMIT`, skip FMP → Yahoo/cache; record state.
- **Unresolvable ticker:** empty search → friendly "no match"; unknown `/ticker/[symbol]` → clear "couldn't resolve" page.
- **Yahoo scrape failures** (401/crumb/429): caught, logged to `provider_state`, degraded to cache.
- **Boundary validation:** Zod on all API route params (query, ticker format); reject malformed input early.
- **Partial comparison:** if the comparison ticker fails but the primary succeeds, render the primary and note the comparison is unavailable.

### 6.8 Testing

- **Unit (Vitest):** indicators (priority; test-first, golden values, `legacy/data/SPY.csv` fixture); provider normalization + fallback decision logic (mocked `fetch`/`yahoo-finance2`); Zod schemas.
- **Integration:** services against a test Postgres (local Docker or Neon branch) — upsert/dedup on `(company_id, date, source)`, freshness logic, budget gating.
- **E2E smoke (Playwright, light):** "NVIDIA" → NVDA → chart renders; compare renders. Golden path only.

### 6.9 SP1 acceptance criteria

- Search "NVIDIA" resolves to NVDA; "NVDA" opens a ticker page.
- Ticker page shows a price chart, period returns, and indicators from cached/provider data.
- `NVDA` vs `AMD` and `AAPL` vs `SPY` comparisons render (overlay + relative return + volatility + drawdown + table).
- Provider failure or exhausted FMP budget degrades to Yahoo/cache with a visible staleness badge — no crash.
- Indicator unit tests pass against golden values.

## 7. SP2 detailed design — News + sentiment + memo

Build second. Usable slice: a ticker page that shows recent deduped news plus a source-linked daily memo and a News Tone meter, generated by Gemini and cached. Verified against 2026 API realities on 2026-05-25 (Finnhub `/company-news` still free at 60/min; Gemini `@google/genai` SDK with `gemini-3.5-flash` stable + structured JSON output; Yahoo Finance RSS `finance.yahoo.com/rss/headline?s=` working, non-commercial only).

### 7.1 Scope & what's deferred

**In scope (all on the existing `/ticker/[symbol]` page):**
- Recent-news ingestion from **Finnhub `/company-news`** (primary) + **Yahoo Finance RSS** (supplement/fallback), deduped and cached.
- A **news table** (time / source / headline / link) — no per-article tone.
- A **source-linked daily memo** + **News Tone meter**, both produced by one batched Gemini call; auto-loaded cache-first with a Regenerate button.

**Deferred (out of SP2):** standalone `/news` page; per-article sentiment / FinBERT; daily tone-trend history & "what changed since yesterday" (needs a scheduler); cross-ticker news filters; retention-pruning job + provider-health page (SP4); embedding/MinHash dedupe.

**Key product decisions (locked during brainstorming):**
- The aggregate News Tone score (0–100) is **emitted directly by Gemini** (label + score + rationale), not computed by a separate deterministic formula.
- The **news table carries no per-article tone**; tone evidence is conveyed by the memo's grouped sections, which cite article IDs.
- The memo is **auto, cache-first**: the page paints instantly with prices + news; the memo card client-fetches after paint and regenerates only when stale; a manual Regenerate button forces a new one. Gemini runs only when stale → low cost.

### 7.2 Data model — two new tables

Add to `lib/db/schema.ts` (imports add `jsonb` from `drizzle-orm/pg-core`):

- **`articles`** — `id` uuid PK · `companyId` uuid FK→`companies.id` · `source` text (`finnhub`|`yahoo_rss`) · `sourceArticleId` text? · `url` text · `urlHash` text (sha256 of canonicalized URL) · `title` text · `summary` text? · `publishedAt` timestamptz · `imageUrl` text? · `related` text? (comma-joined tickers from Finnhub; null for RSS) · `createdAt` timestamptz default now · `expiresAt` timestamptz? (set ~90d out; pruning is SP4). **Unique `(companyId, urlHash)`** → hard cross-source URL dedupe.
- **`daily_memos`** — `id` uuid PK · `companyId` uuid FK · `memoDate` date · `model` text · `summaryJson` jsonb (full structured output, with `source_article_ids` already resolved to article UUIDs) · `toneLabel` text · `toneScore` int (0–100) · `sourceArticleIds` jsonb (union of cited article UUIDs) · `basedOnArticleCount` int · `generatedAt` timestamptz default now. **Unique `(companyId, memoDate)`** → one memo per ticker per day; Regenerate upserts.

**Staleness trigger for regeneration:** no memo exists for `(company, today)` **or** an `articles` row for that company has `createdAt > memo.generatedAt`.

**`provider_state`:** add `finnhub` and `gemini` rows (reuse the existing table for health: `lastSuccessAt`/`lastErrorAt`/`lastError` + `callsToday`). `dailyLimit` stays `notNull`; set Finnhub to a high sentinel (not hard-gated — 60/min is ample for one user, and the ~3h news cache prevents hammering) and Gemini to `GEMINI_DAILY_LIMIT`. FMP stays hard-gated exactly as in SP1.

**`lib/env.ts` additions:** `GEMINI_MODEL` (`z.string().min(1).default("gemini-3.5-flash")`), `GEMINI_PREVIEW_MODEL` (`z.string().min(1).optional()`), `GEMINI_DAILY_LIMIT` (`z.coerce.number().int().positive().default(200)`). `FINNHUB_API_KEY` and `GEMINI_API_KEY` **stay optional** — the features degrade gracefully when a key is absent. News lookback (7 days) and the news cache TTL (~3h) live as code constants, not env.

A migration is generated via `drizzle-kit generate` (same flow as SP1) and applied to Neon.

### 7.3 Provider layer (`lib/providers/`, server-side)

Normalized type added to `lib/types.ts`:

```ts
export interface NewsArticle {
  source: "finnhub" | "yahoo_rss";
  sourceArticleId: string | null;
  url: string;
  title: string;
  summary: string | null;
  publishedAt: Date;
  imageUrl: string | null;
  related: string | null; // comma-joined tickers (Finnhub); null for RSS
}
```

- **`finnhub.ts`** — `companyNews(ticker, fromIso, toIso): Promise<NewsArticle[]>`. `GET https://finnhub.io/api/v1/company-news?symbol=&from=YYYY-MM-DD&to=YYYY-MM-DD`, auth via `X-Finnhub-Token` header. Maps raw `{id, datetime (unix s), headline, source, summary, url, image, related, category}` → `NewsArticle` (`sourceArticleId = String(id)`, `publishedAt = new Date(datetime*1000)`, `title = headline`, `imageUrl = image`). Records success/error to `provider_state`. North-America-only is acceptable (US-listed scope).
- **`yahoo-rss.ts`** — `companyNews(ticker): Promise<NewsArticle[]>`. Fetches `https://finance.yahoo.com/rss/headline?s=TICKER`, parses with **`fast-xml-parser`** (new dep). Maps each `<item>` `{title, link, pubDate, description}` → `NewsArticle` (`source = "yahoo_rss"`, `sourceArticleId = null`, `url = link`, `publishedAt = new Date(pubDate)`, `summary` = HTML-stripped description, `imageUrl = null`). Wrapped in try/catch (scrape-fragile); failures recorded and degraded silently. Returns ~25 most-recent items.
- **`gemini.ts`** — `generateMemo(input: MemoInput): Promise<MemoOutput>` via **`@google/genai`** (new dep). Structured JSON output, low temperature (~0.2), model from env. Records success/error + increments the `gemini` call count. Contract in §7.5.

### 7.4 Services (`lib/services/`)

- **`news-service.ts`** — `getNews(ticker, { force? }): Promise<{ articles: StoredArticle[]; asOf: Date }>`:
  1. Resolve `companyId` (reuse the companies repo).
  2. **Freshness:** if the newest `article.createdAt` for the company is younger than the ~3h TTL and not `force` → serve cache (last 7 days, newest first).
  3. Else fetch **Finnhub** (`from = today−7d`, `to = today`, when the key is present) **+ Yahoo RSS**, merge.
  4. **Dedupe:** canonicalize URL (lowercase host, strip `utm_*`/tracking query params, trim trailing slash) → `urlHash`; the unique constraint collapses exact cross-source dups; additionally soft-skip near-identical normalized titles (lowercase, strip punctuation + source suffixes) within the window.
  5. Upsert (`onConflict (companyId, urlHash) do nothing`), set `expiresAt = now + 90d`; return the 7-day window.
  - **Degrade:** a provider error is caught + recorded; if both providers fail, serve cache; if there's no cache, return an empty list (UI shows "news unavailable").
- **`memo-service.ts`** — `getMemo(ticker, { force? }): Promise<{ memo, status: "ok"|"no_news"|"unavailable"|"error", citedArticles }>`:
  1. **Cache-first:** load `daily_memos` for `(company, today)`; if present, not `force`, and no newer article → return it (with its cited articles).
  2. Else ensure news is fresh (`news-service.getNews`), gather the last-7-day articles. **Zero articles → return `no_news`** (no Gemini call).
  3. If `GEMINI_API_KEY` missing **or** the `GEMINI_DAILY_LIMIT` guardrail is exceeded → return `unavailable`.
  4. Build `MemoInput`: assign each article a short id (`a1…aN`) for citation; add price context (latest close + 1D/5D/1M/1Y returns via `price-service`).
  5. Call `gemini.generateMemo`, validate (§7.5). Map cited short-ids → article UUIDs, applying the citation guards. Upsert `daily_memos` (`onConflict (companyId, memoDate) do update`). Return `ok` with the memo + a `{id, title, url, source}` lookup of cited articles.

### 7.5 Gemini memo contract (trust-critical core)

**`MemoInput`:** `{ ticker, companyName, date, priceContext: { latestClose, currency, returns: {d1, d5, m1, y1} }, articles: [{ id: "a1", source, publishedAt, headline, summary, related }] }`.

**`MemoOutput`** (Zod schema; also the Gemini `responseSchema`):

```ts
const development = z.object({
  claim: z.string(),
  why_it_matters: z.string(),
  source_article_ids: z.array(z.string()),
  confidence: z.enum(["low", "medium", "high"]),
});
const memoOutput = z.object({
  ticker: z.string(),
  date: z.string(),
  one_sentence_takeaway: z.string(),
  bullish_developments: z.array(development),
  bearish_developments: z.array(development),
  neutral_or_operational_updates: z.array(development),
  watch_items: z.array(z.string()),
  caveats: z.array(z.string()),
  overall_news_tone: z.object({
    label: z.enum(["bearish", "somewhat_bearish", "neutral", "somewhat_bullish", "bullish"]),
    score: z.number().int().min(0).max(100),
    rationale: z.string(),
  }),
});
```

(Dropped from `plan.md` §H for SP2: `comparison_context` — single-ticker page; and a speculative `market_reaction` — price context stays factual in the takeaway, no causal guessing.)

**Prompt rules (ported from `plan.md` §H):** never output buy/sell/hold or price targets; every development must cite ≥1 provided article id; use only the supplied headlines/summaries — never invent facts or imply access to full article bodies; separate confirmed company events from analyst speculation; if evidence is thin/duplicated/stale, say so in `caveats`, lower `confidence`, and return mostly-empty arrays; the score is **News Tone** (coverage tone), not an investment outlook; the rationale must reference the actual articles.

**Model config:** `model = env.GEMINI_MODEL` (default `gemini-3.5-flash`; `GEMINI_PREVIEW_MODEL` configurable); `responseMimeType: "application/json"` + `responseSchema`; temperature ~0.2. The exact `@google/genai` config field names (and whether `zod-to-json-schema` is needed) are pinned against the installed SDK version at build time.

**Code-enforced trust guards (not prompt-trusted):**
- Validate with Zod; on failure → one retry, then degrade to `unavailable` (never store malformed memos).
- Map cited short-ids → article UUIDs; **drop any id not in the input set** (hallucination guard); if a development cites zero valid articles after mapping, **drop that development**. The "every claim is source-linked" guarantee is thus enforced in code, not just the prompt.
- Persist `summaryJson` with `source_article_ids` **resolved to article UUIDs** so the UI links each claim straight to its source rows.

### 7.6 API route & data flow (Approach 1 — two endpoints, client-orchestrated, simplified to one new route)

- **Ticker page (server component)** calls `news-service.getNews` + the existing `price-service` and **SSRs** the header, price chart, and **news table** — instant paint (news is cached/fast).
- **Memo card (client component)** fetches **`GET /api/memo/[symbol]`** after paint (spinner). The route is cache-first and generates only when stale. **Regenerate** = same route with `?force=1`. Response = `{ status, memo, citedArticles }`.
- The route Zod-validates the symbol (reuse SP1's ticker regex) and sets **`export const maxDuration = 60`** (news + Gemini can be slow); the page never waits on Gemini.
- No `/api/news` route — news is SSR'd and auto-refreshes on load when older than the TTL.

### 7.7 UI (`components/`, ticker-page additions)

Page order: existing header + price chart → **memo card** → **news table**. A small "Research assistant, not investment advice" note sits near the memo.

- **`memo-card.tsx` (client, centerpiece)** — fetches `/api/memo/[symbol]` on mount. States: `loading` (spinner, "Generating today's memo…"), `ok`, `no_news` ("No recent news in the last 7 days"), `unavailable` ("Memo unavailable — Gemini key missing or daily limit reached"), `error` (friendly retry). `ok` layout: prominent **one-sentence takeaway** → **tone meter** → **Bullish / Bearish / Neutral** sections (each claim shows `why_it_matters`, a confidence tag, and numbered **source chips** that link out to the cited articles) → **Watch items** → **Caveats**. A **Regenerate** button re-fetches with `?force=1`. **Transparency footer:** "Generated by {model} at {time} from {N} sources · every claim links to its source · News Tone reflects coverage tone, not a forecast · not investment advice."
- **`tone-meter.tsx` (presentational)** — 0–100 gauge with the label, colored across the bearish→bullish range; clearly labeled "News Tone." Lives inside the memo card (the score isn't known until the memo loads).
- **`news-table.tsx` (server-rendered)** — columns: **Time** (relative; absolute on hover) · **Source** (finnhub/yahoo + original publisher) · **Headline** (links out, `target="_blank" rel="noopener"`). Newest first, last 7 days. Empty state: "No recent news."

### 7.8 Error handling — "degrade, never crash"

- Missing `FINNHUB_API_KEY` → Yahoo RSS only. Missing `GEMINI_API_KEY` → memo `unavailable`. Neither crashes the page.
- Finnhub error → caught, recorded to `provider_state`, fall back to Yahoo RSS + cache. Yahoo RSS error (404/parse) → caught, recorded, degrade silently. Both fail → serve cached articles (with honest staleness); no cache → "news unavailable" empty state.
- Gemini failure (network/timeout/invalid JSON after one retry) → `error`; card offers retry; nothing malformed stored. `GEMINI_DAILY_LIMIT` exceeded → `unavailable`. Zero articles → `no_news` (no Gemini call).
- Zod-validate the `/api/memo` symbol param; `maxDuration = 60`; overflow → `error` → card retry. Citation guards block hallucinated/unsourced claims. `provider_state` updated for `finnhub`/`gemini` on every call (feeds the SP4 health page).

### 7.9 Testing

- **Unit (Vitest):** Finnhub normalization (unix→Date, field mapping; mocked fetch); Yahoo RSS parsing (sample XML → articles, HTML-strip, empty/bad feed → `[]`); dedupe (URL canonicalization + hash, cross-source collapse, soft title-dedupe); `MemoOutput` Zod schema (valid/invalid) + citation-guard mapping (hallucinated id dropped, zero-cite development dropped); tone-meter score→segment mapping.
- **Integration (live Neon test DB, as SP1):** `news-service` upsert/dedup on `(companyId, urlHash)`, freshness gating, 90d `expiresAt`; `memo-service` cache-first, staleness trigger, `force=1`, `no_news` and `unavailable` paths — **Gemini mocked**.
- **E2E smoke (Playwright, golden path):** ticker page renders the news table and the memo card transitions loading → rendered memo. The memo card's client fetch to **`/api/memo` is intercepted with a fixture** (deterministic, free — never hits live Gemini). The news table is server-rendered, so route interception doesn't apply; it's made deterministic by **seeding the test DB** with a couple of `articles` rows. Finnhub/Yahoo aren't called in the smoke run.
- Lint still OOM-crashes locally on this machine; verified in CI per the standing note.

### 7.10 SP2 acceptance criteria

- A searched ticker (e.g., NVDA) shows a news table populated from Finnhub and/or Yahoo RSS within the last 7 days, with duplicates collapsed.
- The memo card loads cache-first: first visit of the day generates a memo; later visits serve cache; Regenerate forces a fresh one.
- Every memo development links to ≥1 source article; no claim renders without a valid cited article (code-enforced).
- The News Tone meter shows Gemini's 0–100 score + label, is labeled "News Tone," carries the not-advice disclaimer, and no buy/sell/hold text appears anywhere.
- A missing Gemini/Finnhub key or a provider failure degrades to a clear `unavailable`/empty state without crashing; staleness is shown honestly.
- Unit + integration + E2E suites pass with Gemini mocked.

## 8. SP3 detailed design — Fundamentals (SEC EDGAR)

Build third. Usable slice: a stock's ticker page shows a fundamentals card (valuation, profitability, financial health, latest financials) sourced from official SEC filings, cached, with every figure traceable to a 10-K. Verified against live SEC endpoints on 2026-05-25: `company_tickers.json` maps ticker→CIK; `data.sec.gov` requires a contact `User-Agent` (403 without one); `companyfacts` returns current XBRL facts (AAPL: 3.75 MB, 503 us-gaap concepts, all needed concepts present and current to the latest 10-Q).

### 8.1 Scope & what's deferred

**In scope (on the existing `/ticker/[symbol]` page, stocks only):**
- SEC CIK resolution (ticker→CIK via `company_tickers.json`), cached on the company row.
- A single `companyfacts` fetch per company (cold cache only), normalized to ~12 concepts and cached as a small snapshot.
- A **fundamentals card** (client-fetched, mirrors the SP2 memo) showing latest-annual (10-K) figures + the latest balance sheet, with valuation multiples computed from the cached close.

**Deferred (out of SP3):** trailing-twelve-month (TTM) assembly; multi-period history/trends; EV/EBITDA and dividend yield (missing/messy XBRL tags); peer fundamentals comparison; segment data; the deferred paid Sharadar upgrade (§3); retention-pruning of snapshots (SP4).

**Key product decisions (locked during brainstorming):**
- **Data acquisition = one `companyfacts` call**, not per-concept (`companyconcept`) or `frames`. Simplest code, one round-trip, future-proof; the few-MB payload is a non-issue given a multi-day cache (fetch happens at most once per company per TTL window).
- **Period basis = latest annual (10-K).** Flow metrics (revenue, net income, EPS, operating income, gross profit) use the most-recent annual (`fp:"FY"`) fact; balance-sheet metrics + shares use the latest available instant (may be a 10-Q — more current, labeled honestly). TTM deferred (avoids fragile 4-quarter assembly).
- **Metrics = SEC-native + key valuation.** Price-derived Market Cap, P/E, P/S are computed at request time from the already-cached close (no extra provider call) so they never go stale; price-independent ratios derive purely from stored concepts.
- **Render = client-fetched** `/api/fundamentals/[symbol]`, mirroring the memo card so a cold SEC fetch never blocks page paint.

### 8.2 Data model

Add to `lib/db/schema.ts`:

- **`companies`** gains `cik text` (nullable; 10-digit zero-padded string, e.g. `"0000320193"`). Resolved once and stored; reused thereafter.
- **`company_fundamentals`** — one current snapshot per company (refresh upserts): `id` uuid PK · `companyId` uuid FK→`companies.id` **unique** · `conceptsJson` jsonb (normalized reported concepts + per-concept period/filing metadata — *not* the raw 3.75 MB blob) · `fiscalYear` int · `incomePeriodEnd` date (FY period end used for flow metrics) · `balanceSheetAsOf` date (latest balance-sheet instant) · `filingForm` text (`10-K`) · `filedAt` date · `source` text (`sec_edgar`) · `fetchedAt` timestamptz default now · `expiresAt` timestamptz (~7d out). **Unique `(companyId)`.**
- **`provider_state`** gains a `sec` row: high sentinel `dailyLimit` (not hard-gated — SEC has no daily cap, only a ≤10 req/s guideline that hard caching respects); `lastSuccessAt`/`lastErrorAt`/`lastError` feed the SP4 health page.

A migration is generated via `drizzle-kit generate` and applied to Neon (same flow as SP1/SP2).

**`lib/env.ts` addition:** `SEC_USER_AGENT` (`z.string().min(1).optional()`). SEC returns 403 without a contact `User-Agent`; if absent, the feature degrades to `unavailable` (consistent with optional `FINNHUB_API_KEY`/`GEMINI_API_KEY`). `.env.example` documents the format `"AppName contact@email"`.

### 8.3 Provider layer (`lib/providers/sec.ts`, server-side)

- `resolveCik(ticker): Promise<string | null>` — GET `https://www.sec.gov/files/company_tickers.json`, find the entry whose `ticker` matches (case-insensitive), zero-pad `cik_str` to 10 digits. Returns `null` if not found. Sends `User-Agent: env.SEC_USER_AGENT`; records `provider_state(sec)`.
- `fetchCompanyFacts(cik): Promise<RawCompanyFacts | null>` — GET `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` with the User-Agent. Returns parsed JSON or `null` on 403/404/parse error. Records success/error.
- Both return `null` (never throw to the page) when `SEC_USER_AGENT` is missing.

### 8.4 Pure extraction & derivation (`lib/fundamentals/`)

- **`extract.ts`** — `extractConcepts(raw): FundamentalConcepts`. For each logical metric, walk a **fallback list** of XBRL tags and read the right **unit bucket**:
  - Revenue: `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `SalesRevenueNet` (USD)
  - Net Income: `NetIncomeLoss` (USD)
  - EPS: `EarningsPerShareDiluted` → `EarningsPerShareBasic` (USD/shares)
  - Operating Income: `OperatingIncomeLoss` (USD)
  - Gross Profit: `GrossProfit` (USD)
  - Assets / Liabilities / Equity: `Assets`, `Liabilities`, `StockholdersEquity` → `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` (USD)
  - Current Assets / Liabilities: `AssetsCurrent`, `LiabilitiesCurrent` (USD)
  - Shares outstanding: dei `EntityCommonStockSharesOutstanding` (shares)
  - **Period selection:** flow metrics → most-recent fact with `fp === "FY"` (annual); instant/balance-sheet metrics + shares → latest fact by `end`. Each chosen fact's `end`/`form`/`filed`/`fy` is recorded for honest labeling. Missing tag → `null`.
- **`derive.ts`** — `deriveMetrics(concepts, latestClose): FundamentalsView`. Computes: Gross Margin = grossProfit/revenue; ROE = netIncome/equity; ROA = netIncome/assets; Debt/Equity = totalLiabilities/equity; Current Ratio = currentAssets/currentLiabilities; Market Cap = latestClose×shares; P/E = latestClose/dilutedEps; P/S = marketCap/revenue. Every divisor guarded (null/0 → `null`). Pure; no I/O.

Both are pure and unit-tested with a trimmed AAPL `companyfacts` fixture.

### 8.5 Service & API

- **`lib/db/fundamentals.ts`** — `getFundamentalsSnapshot(companyId)`, `upsertFundamentalsSnapshot(...)` (`onConflict (companyId) do update`).
- **`lib/services/fundamentals-service.ts`** — `getFundamentals(ticker): Promise<{ status: "ok"|"not_applicable"|"unavailable"|"error", view, asOf, source }>`:
  1. Resolve company (reuse the companies repo / price-service company path). `assetType !== "stock"` → `not_applicable`.
  2. **Cache-first:** snapshot with `expiresAt > now` → use it.
  3. Else ensure `cik` (resolve + persist on the company if missing). No CIK → `unavailable`.
  4. `fetchCompanyFacts` → `extractConcepts` → upsert snapshot (`expiresAt = now + 7d`). SEC/extract failure → `error`; serve a stale snapshot if one exists; never store malformed.
  5. Read the latest cached close from `price_bars_daily` (no extra provider call) → `deriveMetrics` → return `ok` with the view, `asOf` (period/filing dates), and an EDGAR filings link.
- **`app/api/fundamentals/[symbol]/route.ts`** — `dynamic="force-dynamic"`, `maxDuration=30`; Zod-validate the symbol (reuse the SP1 ticker regex); returns `{ status, view, asOf, source }` at HTTP 200; catch → `{ status:"error", ... }`. The page never waits on SEC.

### 8.6 UI (`components/fundamentals-card.tsx`, ticker-page addition)

Client component; fetches `/api/fundamentals/[symbol]` on mount. States: `loading` (spinner), `ok`, `not_applicable` ("Fundamentals aren't available for ETFs/indexes"), `unavailable` ("Fundamentals unavailable — SEC data couldn't be fetched"), `error` (retry). `ok` groups (each value shows "N/A" when null):
- **Valuation:** Market Cap, P/E (FY EPS), P/S
- **Profitability:** Gross Margin, ROE, ROA, Operating Income
- **Financial Health:** Current Ratio, Debt/Equity, Assets, Liabilities, Equity
- **Latest Financials:** Revenue, Net Income, EPS (diluted)

**Trust footer:** "Source: SEC EDGAR · FY{year} {form} filed {date}; balance sheet as of {date}" + a link to the company's EDGAR filings page. Placed after the memo, before recent news. `lib/types.ts` adds `FundamentalConcepts` and `FundamentalsView`.

### 8.7 Error handling — "degrade, never crash"

- Missing `SEC_USER_AGENT` / SEC 403 / timeout / parse failure → `unavailable`; ETF/index → `not_applicable`; individual missing concepts → "N/A" rows only. Page and card never crash.
- A stale snapshot is served if a fresh fetch fails but an old snapshot exists (honest "as of" labeling). `provider_state(sec)` is updated on every call.

### 8.8 Testing

- **Unit (Vitest):** `extract` (trimmed AAPL fixture → concepts; FY-vs-latest period selection; tag fallback; missing tag → null); `derive` (ratios; divide-by-zero/null → null; margin %, market cap); `resolveCik` (found/not-found vs a small map fixture); route Zod param.
- **Integration (live Neon, SEC mocked):** cache-first, upsert, 7-day TTL staleness, `not_applicable` for an ETF, `unavailable` with no CIK.
- **E2E smoke (Playwright):** the fundamentals card transitions loading→rendered with `/api/fundamentals` intercepted by a fixture.
- **Live verification (standing rule):** real SEC fetch for NVDA + AAPL — confirm figures render sensibly + a screenshot; mocks can't catch real concept-tagging quirks. Lint stays verified in CI (OOM locally).

### 8.9 SP3 acceptance criteria

- A stock ticker (e.g. NVDA, AAPL) shows a fundamentals card with valuation, profitability, financial-health, and latest-financials values sourced from SEC EDGAR, cached after first load.
- Every figure traces to a filing: the card shows the fiscal year, form, filing date, and balance-sheet date, and links to EDGAR.
- Market Cap / P/E / P/S are computed from the latest cached close (no extra provider call) and update as prices refresh.
- An ETF/index (SPY/QQQ) shows `not_applicable`; a missing key/CIK or SEC failure shows `unavailable`; missing individual concepts show "N/A" — no crash.
- Unit + integration + E2E suites pass with SEC mocked.

## 9. Out of scope (MVP)

Per `plan.md` non-goals: no scraping of paywalled article bodies, no trading execution, no portfolio optimization, no public redistribution of provider data, no delisted-company database, no buy/sell/hold output. Also out of scope for the MVP specifically: local FinBERT, a separate Python service, Alpha Vantage, paid data tiers, and a background scheduler.

## 10. Assumptions

- Personal/non-commercial use; app stays private behind a password gate.
- US-listed securities (FMP free is US-only; SEC is US-only).
- Daily (EOD) data is sufficient; no realtime/intraday in the MVP.
- Free-tier quotas (FMP 250/day, Gemini free tier, Finnhub 60/min) are adequate for one user at personal volume with caching.
