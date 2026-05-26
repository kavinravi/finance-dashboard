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
2. **SP2 — News + Sentiment + Memo:** Finnhub + Yahoo RSS ingestion, dedupe, Gemini News Tone + source-linked daily memo, news table + tone meter + memo UI.
3. **SP3 — Fundamentals:** SEC EDGAR CIK mapping + CompanyFacts (minimal concepts), fundamentals card. Adds `cik` to `companies`.
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

## 7. Out of scope (MVP)

Per `plan.md` non-goals: no scraping of paywalled article bodies, no trading execution, no portfolio optimization, no public redistribution of provider data, no delisted-company database, no buy/sell/hold output. Also out of scope for the MVP specifically: local FinBERT, a separate Python service, Alpha Vantage, paid data tiers, and a background scheduler.

## 8. Assumptions

- Personal/non-commercial use; app stays private behind a password gate.
- US-listed securities (FMP free is US-only; SEC is US-only).
- Daily (EOD) data is sufficient; no realtime/intraday in the MVP.
- Free-tier quotas (FMP 250/day, Gemini free tier, Finnhub 60/min) are adequate for one user at personal volume with caching.
