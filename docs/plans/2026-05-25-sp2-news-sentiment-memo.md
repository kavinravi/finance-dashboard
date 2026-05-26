# SP2 — News + Sentiment + Memo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add recent-news ingestion (Finnhub + Yahoo RSS), a source-linked Gemini daily memo, and a News Tone meter to the existing ticker page.

**Architecture:** Two new providers (`finnhub`, `yahoo-rss`) + a Gemini provider feed two services — `news-service` (fetch → dedupe → cache) and `memo-service` (cache-first generate). The ticker page server-renders prices + a news table instantly; a client `MemoCard` fetches `GET /api/memo/[symbol]` after paint (cache-first, regenerates only when stale). Every memo claim is source-linked, enforced in code. Degrade-never-crash throughout.

**Tech Stack:** Next.js 16 App Router, TypeScript, Drizzle + Neon Postgres, Zod, `@google/genai`, `fast-xml-parser`, Vitest, Playwright.

**Authoritative spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §7. Read it before starting.

**Standing notes for this repo:**
- Commits/docs/branches/PRs must contain **no AI-authorship traces** (no "Claude", "Anthropic", "Co-Authored-By", tool names, 🤖). Author stays `kavinravi`.
- `pnpm lint`/eslint OOM-crashes locally and Next 16 doesn't lint on `build` — verify lint in CI, don't chase it locally.
- Unit/service tests live co-located in `lib/**/*.test.ts` and mock the DB repos + providers (see `lib/services/price-service.test.ts` for the exact `vi.hoisted` pattern). DB repos have no dedicated unit tests — they're covered by service mocks + E2E.
- Run a single test file: `pnpm test <path>`. Full suite: `pnpm test`. Typecheck: `pnpm exec tsc --noEmit`.
- `.env` already holds `DATABASE_URL`, `FMP_API_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`. Migrations: `pnpm db:generate` then `pnpm db:migrate`.

---

## File Structure

**New files**
- `lib/providers/finnhub.ts` — Finnhub `/company-news` client + pure `parseFinnhubNews`.
- `lib/providers/finnhub.test.ts` — parser unit tests.
- `lib/providers/yahoo-rss.ts` — Yahoo RSS client + pure `parseYahooRss`.
- `lib/providers/yahoo-rss.test.ts` — parser unit tests.
- `lib/providers/gemini.ts` — `memoOutputSchema` (Zod), `buildPrompt`, `generateMemo`.
- `lib/providers/gemini.test.ts` — schema + prompt unit tests.
- `lib/news/dedupe.ts` — `canonicalizeUrl`, `urlHash`, `normalizeTitle`, `dedupeArticles`.
- `lib/news/dedupe.test.ts` — dedupe unit tests.
- `lib/db/articles.ts` — articles repo.
- `lib/db/daily-memos.ts` — daily-memos repo.
- `lib/services/news-service.ts` — `getNews`.
- `lib/services/news-service.test.ts` — service unit tests (mocked).
- `lib/services/memo-service.ts` — `getMemo`.
- `lib/services/memo-service.test.ts` — service unit tests (mocked).
- `lib/tone.ts` — `toneLabelText`, `toneColor` (pure).
- `lib/tone.test.ts` — tone unit tests.
- `app/api/memo/[symbol]/route.ts` — memo endpoint.
- `components/tone-meter.tsx` — presentational meter.
- `components/news-table.tsx` — server-rendered news table.
- `components/memo-card.tsx` — client memo card.

**Modified files**
- `lib/env.ts` — add `GEMINI_MODEL`, `GEMINI_PREVIEW_MODEL`, `GEMINI_DAILY_LIMIT`.
- `lib/db/schema.ts` — add `articles` + `dailyMemos` tables.
- `lib/types.ts` — add `NewsArticle`.
- `app/ticker/[symbol]/page.tsx` — render `NewsTable` + `MemoCard`.
- `tests/e2e/smoke.spec.ts` — add a memo/news smoke test.
- `package.json` — add `@google/genai`, `fast-xml-parser` (via `pnpm add`).

---

## Task 1: Env additions

**Files:**
- Modify: `lib/env.ts`
- Test: `lib/env.test.ts` (create)

- [ ] **Step 1: Write the failing test**

Create `lib/env.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { env } from "./env";

describe("env (SP2 additions)", () => {
  it("defaults the Gemini model and daily limit", () => {
    expect(env.GEMINI_MODEL).toBe("gemini-3.5-flash");
    expect(env.GEMINI_DAILY_LIMIT).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/env.test.ts`
Expected: FAIL — `env.GEMINI_MODEL` is `undefined`.

- [ ] **Step 3: Add the fields**

In `lib/env.ts`, extend the schema and the parsed object:

```ts
const schema = z.object({
  DATABASE_URL: z.string().url(),
  FMP_API_KEY: z.string().min(1),
  FMP_DAILY_LIMIT: z.coerce.number().int().positive().default(250),
  FINNHUB_API_KEY: z.string().min(1).optional(),
  GEMINI_API_KEY: z.string().min(1).optional(),
  GEMINI_MODEL: z.string().min(1).default("gemini-3.5-flash"),
  GEMINI_PREVIEW_MODEL: z.string().min(1).optional(),
  GEMINI_DAILY_LIMIT: z.coerce.number().int().positive().default(200),
});

const parsed = schema.safeParse({
  DATABASE_URL: process.env.DATABASE_URL,
  FMP_API_KEY: process.env.FMP_API_KEY,
  FMP_DAILY_LIMIT: process.env.FMP_DAILY_LIMIT,
  FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
  GEMINI_API_KEY: process.env.GEMINI_API_KEY,
  GEMINI_MODEL: process.env.GEMINI_MODEL,
  GEMINI_PREVIEW_MODEL: process.env.GEMINI_PREVIEW_MODEL,
  GEMINI_DAILY_LIMIT: process.env.GEMINI_DAILY_LIMIT,
});
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pnpm test lib/env.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/env.ts lib/env.test.ts
git commit -m "Add Gemini env vars for SP2"
```

---

## Task 2: Schema + migration (articles, daily_memos)

**Files:**
- Modify: `lib/db/schema.ts`
- Create (generated): `drizzle/*.sql`

- [ ] **Step 1: Add the tables**

In `lib/db/schema.ts`, add `jsonb` to the import list from `drizzle-orm/pg-core`, then append:

```ts
export const articles = pgTable("articles", {
  id: uuid("id").primaryKey().defaultRandom(),
  companyId: uuid("company_id").notNull().references(() => companies.id),
  source: text("source").notNull(),            // finnhub | yahoo_rss
  sourceArticleId: text("source_article_id"),
  url: text("url").notNull(),
  urlHash: text("url_hash").notNull(),          // sha256 of canonicalized URL
  title: text("title").notNull(),
  summary: text("summary"),
  publishedAt: timestamp("published_at", { withTimezone: true }).notNull(),
  imageUrl: text("image_url"),
  related: text("related"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }),
}, (t) => [unique("uq_article_company_urlhash").on(t.companyId, t.urlHash)]);

export const dailyMemos = pgTable("daily_memos", {
  id: uuid("id").primaryKey().defaultRandom(),
  companyId: uuid("company_id").notNull().references(() => companies.id),
  memoDate: date("memo_date").notNull(),
  model: text("model").notNull(),
  summaryJson: jsonb("summary_json").notNull(),
  toneLabel: text("tone_label").notNull(),
  toneScore: integer("tone_score").notNull(),
  sourceArticleIds: jsonb("source_article_ids").notNull(),
  basedOnArticleCount: integer("based_on_article_count").notNull(),
  generatedAt: timestamp("generated_at", { withTimezone: true }).defaultNow().notNull(),
}, (t) => [unique("uq_memo_company_date").on(t.companyId, t.memoDate)]);
```

- [ ] **Step 2: Generate the migration**

Run: `pnpm db:generate`
Expected: a new `drizzle/0001_*.sql` file containing `CREATE TABLE "articles"` and `CREATE TABLE "daily_memos"`.

- [ ] **Step 3: Verify typecheck**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Apply the migration to Neon**

Run: `pnpm db:migrate`
Expected: applies cleanly (`[✓] migrations applied`).

- [ ] **Step 5: Commit**

```bash
git add lib/db/schema.ts drizzle/
git commit -m "Add articles and daily_memos tables"
```

---

## Task 3: NewsArticle type + Finnhub provider

**Files:**
- Modify: `lib/types.ts`
- Create: `lib/providers/finnhub.ts`
- Test: `lib/providers/finnhub.test.ts`

- [ ] **Step 1: Add the NewsArticle type**

Append to `lib/types.ts`:

```ts
export type NewsArticle = {
  source: "finnhub" | "yahoo_rss";
  sourceArticleId: string | null;
  url: string;
  title: string;
  summary: string | null;
  publishedAt: Date;
  imageUrl: string | null;
  related: string | null; // comma-joined tickers (Finnhub); null for RSS
};
```

- [ ] **Step 2: Write the failing test**

Create `lib/providers/finnhub.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { parseFinnhubNews } from "./finnhub";

describe("parseFinnhubNews", () => {
  it("maps raw items, converting unix seconds to a Date", () => {
    const raw = [{
      id: 12345, datetime: 1716595200, headline: "Acme beats earnings",
      source: "MarketWatch", summary: "Strong quarter.", url: "https://ex.com/a",
      image: "https://ex.com/a.jpg", related: "ACME", category: "company",
    }];
    const out = parseFinnhubNews(raw);
    expect(out).toEqual([{
      source: "finnhub", sourceArticleId: "12345", url: "https://ex.com/a",
      title: "Acme beats earnings", summary: "Strong quarter.",
      publishedAt: new Date(1716595200 * 1000), imageUrl: "https://ex.com/a.jpg", related: "ACME",
    }]);
  });

  it("drops items missing headline or url and blanks empty optional fields", () => {
    const raw = [
      { id: 1, datetime: 1, headline: "", url: "https://x" },
      { id: 2, datetime: 2, headline: "Has title", url: "", summary: "" },
      { id: 3, datetime: 3, headline: "Keep", url: "https://y", summary: "  ", image: "", related: "" },
    ];
    const out = parseFinnhubNews(raw);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ title: "Keep", summary: null, imageUrl: null, related: null });
  });
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pnpm test lib/providers/finnhub.test.ts`
Expected: FAIL — `parseFinnhubNews` not defined.

- [ ] **Step 4: Implement the provider**

Create `lib/providers/finnhub.ts`:

```ts
import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { NewsArticle } from "@/lib/types";

const BASE = "https://finnhub.io/api/v1";
// Not hard-gated — Finnhub free is 60/min with no daily cap, ample for one user.
// The value only feeds the provider_state health row.
export const FINNHUB_DAILY_LIMIT = 100000;

function nonEmpty(v: unknown): string | null {
  return typeof v === "string" && v.trim() !== "" ? v : null;
}

export function parseFinnhubNews(raw: any[]): NewsArticle[] {
  return (raw ?? [])
    .filter((r) => r && r.headline && r.url)
    .map((r) => ({
      source: "finnhub" as const,
      sourceArticleId: r.id != null ? String(r.id) : null,
      url: String(r.url),
      title: String(r.headline),
      summary: nonEmpty(r.summary),
      publishedAt: new Date(Number(r.datetime) * 1000),
      imageUrl: nonEmpty(r.image),
      related: nonEmpty(r.related),
    }));
}

export const finnhub = {
  name: "finnhub",
  async companyNews(ticker: string, fromIso: string, toIso: string): Promise<NewsArticle[]> {
    if (!env.FINNHUB_API_KEY) return [];
    const url = `${BASE}/company-news?symbol=${encodeURIComponent(ticker)}&from=${fromIso}&to=${toIso}`;
    try {
      const res = await fetch(url, { headers: { "X-Finnhub-Token": env.FINNHUB_API_KEY } });
      if (!res.ok) throw new Error(`Finnhub ${res.status}`);
      const data = parseFinnhubNews(await res.json());
      await recordSuccess("finnhub", FINNHUB_DAILY_LIMIT);
      return data;
    } catch (e) {
      await recordError("finnhub", FINNHUB_DAILY_LIMIT, String(e));
      return []; // degrade — never throw
    }
  },
};
```

- [ ] **Step 5: Run it to verify it passes**

Run: `pnpm test lib/providers/finnhub.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add lib/types.ts lib/providers/finnhub.ts lib/providers/finnhub.test.ts
git commit -m "Add Finnhub company-news provider"
```

---

## Task 4: Yahoo RSS provider

**Files:**
- Create: `lib/providers/yahoo-rss.ts`
- Test: `lib/providers/yahoo-rss.test.ts`
- Modify: `package.json` (add `fast-xml-parser`)

- [ ] **Step 1: Add the dependency**

Run: `pnpm add fast-xml-parser`
Expected: `fast-xml-parser` appears in `package.json` dependencies.

- [ ] **Step 2: Write the failing test**

Create `lib/providers/yahoo-rss.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { parseYahooRss } from "./yahoo-rss";

const xml = `<?xml version="1.0"?><rss version="2.0"><channel>
  <item><title>Acme &amp; Co rallies</title><link>https://ex.com/1</link>
    <pubDate>Mon, 25 May 2026 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;Shares up &lt;b&gt;5%&lt;/b&gt;.&lt;/p&gt;</description></item>
  <item><title>Second story</title><link>https://ex.com/2</link>
    <pubDate>Mon, 25 May 2026 09:00:00 GMT</pubDate></item>
</channel></rss>`;

describe("parseYahooRss", () => {
  it("maps items, strips HTML, and decodes entities", () => {
    const out = parseYahooRss(xml);
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({
      source: "yahoo_rss", sourceArticleId: null, url: "https://ex.com/1",
      title: "Acme & Co rallies", summary: "Shares up 5%.", imageUrl: null, related: null,
    });
    expect(out[0].publishedAt instanceof Date).toBe(true);
    expect(out[1].summary).toBeNull();
  });

  it("returns [] for empty or malformed feeds", () => {
    expect(parseYahooRss("<rss><channel></channel></rss>")).toEqual([]);
    expect(parseYahooRss("not xml")).toEqual([]);
  });
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pnpm test lib/providers/yahoo-rss.test.ts`
Expected: FAIL — `parseYahooRss` not defined.

- [ ] **Step 4: Implement the provider**

Create `lib/providers/yahoo-rss.ts`:

```ts
import { XMLParser } from "fast-xml-parser";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { NewsArticle } from "@/lib/types";

const YAHOO_RSS_LIMIT = 100000; // health-row only; not gated
const parser = new XMLParser({ ignoreAttributes: true, htmlEntities: true });

function stripHtml(s: string): string {
  return s.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
}

export function parseYahooRss(xml: string): NewsArticle[] {
  let doc: any;
  try { doc = parser.parse(xml); } catch { return []; }
  const items = doc?.rss?.channel?.item;
  const arr = Array.isArray(items) ? items : items ? [items] : [];
  return arr
    .filter((it: any) => it && it.title && it.link)
    .map((it: any) => {
      const summary = it.description != null ? stripHtml(String(it.description)) : "";
      return {
        source: "yahoo_rss" as const,
        sourceArticleId: null,
        url: String(it.link),
        title: stripHtml(String(it.title)),
        summary: summary === "" ? null : summary,
        publishedAt: it.pubDate ? new Date(String(it.pubDate)) : new Date(),
        imageUrl: null,
        related: null,
      };
    });
}

export const yahooRss = {
  name: "yahoo_rss",
  async companyNews(ticker: string): Promise<NewsArticle[]> {
    const url = `https://finance.yahoo.com/rss/headline?s=${encodeURIComponent(ticker)}`;
    try {
      const res = await fetch(url, { headers: { Accept: "application/rss+xml, application/xml" } });
      if (!res.ok) throw new Error(`Yahoo RSS ${res.status}`);
      const data = parseYahooRss(await res.text());
      await recordSuccess("yahoo_rss", YAHOO_RSS_LIMIT);
      return data;
    } catch (e) {
      await recordError("yahoo_rss", YAHOO_RSS_LIMIT, String(e));
      return []; // scrape-fragile — degrade silently
    }
  },
};
```

- [ ] **Step 5: Run it to verify it passes**

Run: `pnpm test lib/providers/yahoo-rss.test.ts`
Expected: PASS (2 tests). If entity decoding differs, confirm `htmlEntities: true` is set on the parser.

- [ ] **Step 6: Commit**

```bash
git add package.json pnpm-lock.yaml lib/providers/yahoo-rss.ts lib/providers/yahoo-rss.test.ts
git commit -m "Add Yahoo Finance RSS news provider"
```

---

## Task 5: Dedupe utilities

**Files:**
- Create: `lib/news/dedupe.ts`
- Test: `lib/news/dedupe.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/news/dedupe.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { canonicalizeUrl, urlHash, normalizeTitle, dedupeArticles } from "./dedupe";
import type { NewsArticle } from "@/lib/types";

describe("canonicalizeUrl", () => {
  it("lowercases host, drops www, strips tracking params and trailing slash", () => {
    expect(canonicalizeUrl("https://WWW.Ex.com/a/?utm_source=x&id=5#frag"))
      .toBe("https://ex.com/a?id=5");
    expect(canonicalizeUrl("https://ex.com/a/")).toBe("https://ex.com/a");
  });
  it("falls back to the trimmed string for non-URLs", () => {
    expect(canonicalizeUrl("  not a url ")).toBe("not a url");
  });
});

describe("urlHash", () => {
  it("is stable for URLs that canonicalize equally", () => {
    expect(urlHash("https://www.ex.com/a?utm_medium=y")).toBe(urlHash("https://ex.com/a"));
  });
});

describe("normalizeTitle", () => {
  it("lowercases and strips punctuation", () => {
    expect(normalizeTitle("Acme, Inc. Beats!")).toBe("acme inc beats");
  });
});

const mk = (over: Partial<NewsArticle>): NewsArticle => ({
  source: "finnhub", sourceArticleId: null, url: "https://ex.com/x", title: "T",
  summary: null, publishedAt: new Date("2026-05-25T00:00:00Z"), imageUrl: null, related: null, ...over,
});

describe("dedupeArticles", () => {
  it("collapses same canonical URL across sources, keeping the newest", () => {
    const out = dedupeArticles([
      mk({ source: "finnhub", url: "https://ex.com/a", publishedAt: new Date("2026-05-25T08:00:00Z") }),
      mk({ source: "yahoo_rss", url: "https://www.ex.com/a/?utm_source=z", publishedAt: new Date("2026-05-25T10:00:00Z") }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].source).toBe("yahoo_rss"); // newer kept
  });
  it("collapses near-identical titles with different URLs", () => {
    const out = dedupeArticles([
      mk({ url: "https://a.com/1", title: "Acme beats earnings!" }),
      mk({ url: "https://b.com/2", title: "Acme Beats Earnings" }),
    ]);
    expect(out).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/news/dedupe.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `lib/news/dedupe.ts`:

```ts
import { createHash } from "node:crypto";
import type { NewsArticle } from "@/lib/types";

const TRACKING = new Set(["fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"]);

export function canonicalizeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    u.hash = "";
    u.hostname = u.hostname.toLowerCase().replace(/^www\./, "");
    for (const k of [...u.searchParams.keys()]) {
      if (/^utm_/i.test(k) || TRACKING.has(k.toLowerCase())) u.searchParams.delete(k);
    }
    return u.toString().replace(/\/$/, "");
  } catch {
    return raw.trim();
  }
}

export function urlHash(raw: string): string {
  return createHash("sha256").update(canonicalizeUrl(raw)).digest("hex");
}

export function normalizeTitle(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
}

export function dedupeArticles(articles: NewsArticle[]): NewsArticle[] {
  const seenHash = new Set<string>();
  const seenTitle = new Set<string>();
  const out: NewsArticle[] = [];
  const sorted = [...articles].sort((a, b) => b.publishedAt.getTime() - a.publishedAt.getTime());
  for (const a of sorted) {
    const h = urlHash(a.url);
    const t = normalizeTitle(a.title);
    if (seenHash.has(h) || (t.length > 0 && seenTitle.has(t))) continue;
    seenHash.add(h);
    if (t.length > 0) seenTitle.add(t);
    out.push(a);
  }
  return out;
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pnpm test lib/news/dedupe.test.ts`
Expected: PASS (6 assertions across the describes).

- [ ] **Step 5: Commit**

```bash
git add lib/news/dedupe.ts lib/news/dedupe.test.ts
git commit -m "Add news dedupe utilities"
```

---

## Task 6: DB repos (articles, daily_memos)

**Files:**
- Create: `lib/db/articles.ts`
- Create: `lib/db/daily-memos.ts`

(No dedicated unit tests — these are exercised by the service tests in Tasks 7 and 9, matching the SP1 convention where `price-bars.ts`/`companies.ts` have none.)

- [ ] **Step 1: Implement the articles repo**

Create `lib/db/articles.ts`:

```ts
import { db } from "./client";
import { articles } from "./schema";
import { and, eq, gte, gt, desc, inArray } from "drizzle-orm";
import { urlHash } from "@/lib/news/dedupe";
import type { NewsArticle } from "@/lib/types";

export type ArticleRow = typeof articles.$inferSelect;

const RETENTION_DAYS = 90;

export async function upsertArticles(companyId: string, items: NewsArticle[]): Promise<void> {
  if (items.length === 0) return;
  const expiresAt = new Date(Date.now() + RETENTION_DAYS * 86_400_000);
  const rows = items.map((a) => ({
    companyId, source: a.source, sourceArticleId: a.sourceArticleId, url: a.url,
    urlHash: urlHash(a.url), title: a.title, summary: a.summary,
    publishedAt: a.publishedAt, imageUrl: a.imageUrl, related: a.related, expiresAt,
  }));
  for (let i = 0; i < rows.length; i += 500) {
    await db.insert(articles).values(rows.slice(i, i + 500)).onConflictDoNothing();
  }
}

export async function getRecentArticles(companyId: string, sinceIso: string): Promise<ArticleRow[]> {
  return db.select().from(articles)
    .where(and(eq(articles.companyId, companyId), gte(articles.publishedAt, new Date(sinceIso))))
    .orderBy(desc(articles.publishedAt));
}

export async function newestArticleCreatedAt(companyId: string): Promise<Date | null> {
  const [row] = await db.select().from(articles)
    .where(eq(articles.companyId, companyId)).orderBy(desc(articles.createdAt)).limit(1);
  return row?.createdAt ?? null;
}

export async function hasArticleNewerThan(companyId: string, t: Date): Promise<boolean> {
  const [row] = await db.select().from(articles)
    .where(and(eq(articles.companyId, companyId), gt(articles.createdAt, t))).limit(1);
  return !!row;
}

export async function getArticlesByIds(ids: string[]): Promise<ArticleRow[]> {
  if (ids.length === 0) return [];
  return db.select().from(articles).where(inArray(articles.id, ids));
}
```

- [ ] **Step 2: Implement the daily-memos repo**

Create `lib/db/daily-memos.ts`:

```ts
import { db } from "./client";
import { dailyMemos } from "./schema";
import { and, eq } from "drizzle-orm";

export type DailyMemoRow = typeof dailyMemos.$inferSelect;

export async function getMemoForDate(companyId: string, memoDate: string): Promise<DailyMemoRow | undefined> {
  const [row] = await db.select().from(dailyMemos)
    .where(and(eq(dailyMemos.companyId, companyId), eq(dailyMemos.memoDate, memoDate)));
  return row;
}

export async function upsertMemo(row: {
  companyId: string; memoDate: string; model: string; summaryJson: unknown;
  toneLabel: string; toneScore: number; sourceArticleIds: string[]; basedOnArticleCount: number;
}): Promise<void> {
  await db.insert(dailyMemos)
    .values({ ...row, generatedAt: new Date() })
    .onConflictDoUpdate({
      target: [dailyMemos.companyId, dailyMemos.memoDate],
      set: {
        model: row.model, summaryJson: row.summaryJson, toneLabel: row.toneLabel,
        toneScore: row.toneScore, sourceArticleIds: row.sourceArticleIds,
        basedOnArticleCount: row.basedOnArticleCount, generatedAt: new Date(),
      },
    });
}
```

- [ ] **Step 3: Verify typecheck**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add lib/db/articles.ts lib/db/daily-memos.ts
git commit -m "Add articles and daily-memos repositories"
```

---

## Task 7: news-service

**Files:**
- Create: `lib/services/news-service.ts`
- Test: `lib/services/news-service.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/services/news-service.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  getCompanyByTicker, upsertArticles, getRecentArticles, newestArticleCreatedAt,
  finnhubNews, yahooRssNews,
} = vi.hoisted(() => ({
  getCompanyByTicker: vi.fn(),
  upsertArticles: vi.fn(),
  getRecentArticles: vi.fn(),
  newestArticleCreatedAt: vi.fn(),
  finnhubNews: vi.fn(),
  yahooRssNews: vi.fn(),
}));

vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker }));
vi.mock("@/lib/db/articles", () => ({ upsertArticles, getRecentArticles, newestArticleCreatedAt }));
vi.mock("@/lib/providers/finnhub", () => ({ finnhub: { companyNews: finnhubNews } }));
vi.mock("@/lib/providers/yahoo-rss", () => ({ yahooRss: { companyNews: yahooRssNews } }));

import { getNews } from "./news-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", currency: "USD" };
const article = (url: string, when: string) => ({
  source: "finnhub", sourceArticleId: "1", url, title: "T " + url, summary: null,
  publishedAt: new Date(when), imageUrl: null, related: null,
});

beforeEach(() => {
  [getCompanyByTicker, upsertArticles, getRecentArticles, newestArticleCreatedAt, finnhubNews, yahooRssNews]
    .forEach((m) => m.mockReset());
  getCompanyByTicker.mockResolvedValue(company);
  getRecentArticles.mockResolvedValue([]);
  finnhubNews.mockResolvedValue([]);
  yahooRssNews.mockResolvedValue([]);
});

describe("getNews", () => {
  it("returns empty when the company is unknown", async () => {
    getCompanyByTicker.mockResolvedValue(undefined);
    const out = await getNews("ZZZZ");
    expect(out.articles).toEqual([]);
    expect(finnhubNews).not.toHaveBeenCalled();
  });

  it("serves cache without calling providers when news is fresh", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date()); // just now → fresh
    await getNews("NVDA");
    expect(finnhubNews).not.toHaveBeenCalled();
    expect(yahooRssNews).not.toHaveBeenCalled();
  });

  it("fetches, dedupes, and upserts when stale", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date(Date.now() - 5 * 3600_000)); // 5h → stale
    finnhubNews.mockResolvedValue([article("https://ex.com/a", "2026-05-25T08:00:00Z")]);
    yahooRssNews.mockResolvedValue([article("https://www.ex.com/a/?utm_source=z", "2026-05-25T10:00:00Z")]);
    await getNews("NVDA");
    expect(upsertArticles).toHaveBeenCalledTimes(1);
    expect(upsertArticles.mock.calls[0][1]).toHaveLength(1); // deduped to one
  });

  it("force-refreshes even when fresh", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date());
    await getNews("NVDA", { force: true });
    expect(finnhubNews).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/services/news-service.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `lib/services/news-service.ts`:

```ts
import { getCompanyByTicker } from "@/lib/db/companies";
import { upsertArticles, getRecentArticles, newestArticleCreatedAt, type ArticleRow } from "@/lib/db/articles";
import { finnhub } from "@/lib/providers/finnhub";
import { yahooRss } from "@/lib/providers/yahoo-rss";
import { dedupeArticles } from "@/lib/news/dedupe";

const LOOKBACK_DAYS = 7;
const CACHE_TTL_MS = 3 * 60 * 60 * 1000; // 3h

function isoDaysAgo(n: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}

export async function getNews(
  ticker: string,
  opts: { force?: boolean } = {},
): Promise<{ articles: ArticleRow[]; asOf: Date | null }> {
  const company = await getCompanyByTicker(ticker);
  if (!company) return { articles: [], asOf: null };

  const newest = await newestArticleCreatedAt(company.id);
  const fresh = newest !== null && Date.now() - newest.getTime() < CACHE_TTL_MS;

  if (opts.force || !fresh) {
    const fromIso = isoDaysAgo(LOOKBACK_DAYS);
    const toIso = new Date().toISOString().slice(0, 10);
    const [fh, yr] = await Promise.all([
      finnhub.companyNews(company.ticker, fromIso, toIso), // each degrades to [] internally
      yahooRss.companyNews(company.ticker),
    ]);
    await upsertArticles(company.id, dedupeArticles([...fh, ...yr]));
  }

  const rows = await getRecentArticles(company.id, isoDaysAgo(LOOKBACK_DAYS));
  return { articles: rows, asOf: rows[0]?.publishedAt ?? null };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pnpm test lib/services/news-service.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/services/news-service.ts lib/services/news-service.test.ts
git commit -m "Add news-service (fetch, dedupe, cache)"
```

---

## Task 8: Gemini provider

**Files:**
- Create: `lib/providers/gemini.ts`
- Test: `lib/providers/gemini.test.ts`
- Modify: `package.json` (add `@google/genai`)

- [ ] **Step 1: Add the dependency**

Run: `pnpm add @google/genai`
Expected: `@google/genai` appears in `package.json` dependencies.

- [ ] **Step 2: Write the failing test**

Create `lib/providers/gemini.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { memoOutputSchema, buildPrompt, type MemoInput } from "./gemini";

const validMemo = {
  ticker: "NVDA", date: "2026-05-25", one_sentence_takeaway: "Quiet week.",
  bullish_developments: [], bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: [], caveats: ["Sparse coverage."],
  overall_news_tone: { label: "neutral", score: 50, rationale: "Few articles." },
};

describe("memoOutputSchema", () => {
  it("accepts a valid memo", () => {
    expect(memoOutputSchema.safeParse(validMemo).success).toBe(true);
  });
  it("rejects an out-of-range score and a bad label", () => {
    expect(memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "neutral", score: 150, rationale: "x" } }).success).toBe(false);
    expect(memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "great", score: 50, rationale: "x" } }).success).toBe(false);
  });
});

describe("buildPrompt", () => {
  it("includes every article id and forbids buy/sell/hold", () => {
    const input: MemoInput = {
      ticker: "NVDA", companyName: "NVIDIA", date: "2026-05-25",
      priceContext: { latestClose: 215.3, currency: "USD", returns: { d1: -0.019, d5: 0.02, m1: 0.05, y1: 0.62 } },
      articles: [
        { id: "a1", source: "finnhub", publishedAt: "2026-05-24T12:00:00Z", headline: "Chip demand", summary: "Up", related: "NVDA" },
        { id: "a2", source: "yahoo_rss", publishedAt: "2026-05-23T12:00:00Z", headline: "Supply news", summary: null, related: null },
      ],
    };
    const p = buildPrompt(input);
    expect(p).toContain("[a1]");
    expect(p).toContain("[a2]");
    expect(p.toLowerCase()).toContain("never output buy");
  });
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pnpm test lib/providers/gemini.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement**

Create `lib/providers/gemini.ts`. **Note:** the structured-output approach here uses `responseMimeType: "application/json"` plus an explicit shape in the prompt, then Zod-validates with one retry — this avoids JSON-Schema-translation fragility. If you later tighten it with `responseSchema`/`responseJsonSchema`, verify the exact field name against the installed `@google/genai` types (`node_modules/@google/genai`).

```ts
import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";

export const developmentSchema = z.object({
  claim: z.string(),
  why_it_matters: z.string(),
  source_article_ids: z.array(z.string()),
  confidence: z.enum(["low", "medium", "high"]),
});

export const memoOutputSchema = z.object({
  ticker: z.string(),
  date: z.string(),
  one_sentence_takeaway: z.string(),
  bullish_developments: z.array(developmentSchema),
  bearish_developments: z.array(developmentSchema),
  neutral_or_operational_updates: z.array(developmentSchema),
  watch_items: z.array(z.string()),
  caveats: z.array(z.string()),
  overall_news_tone: z.object({
    label: z.enum(["bearish", "somewhat_bearish", "neutral", "somewhat_bullish", "bullish"]),
    score: z.number().int().min(0).max(100),
    rationale: z.string(),
  }),
});
export type MemoOutput = z.infer<typeof memoOutputSchema>;

export type MemoInputArticle = {
  id: string; source: string; publishedAt: string; headline: string; summary: string | null; related: string | null;
};
export type MemoInput = {
  ticker: string; companyName: string; date: string;
  priceContext: {
    latestClose: number | null; currency: string | null;
    returns: { d1: number | null; d5: number | null; m1: number | null; y1: number | null };
  };
  articles: MemoInputArticle[];
};

const SHAPE_EXAMPLE = {
  ticker: "TICK", date: "YYYY-MM-DD", one_sentence_takeaway: "string",
  bullish_developments: [{ claim: "string", why_it_matters: "string", source_article_ids: ["a1"], confidence: "low|medium|high" }],
  bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: ["string"], caveats: ["string"],
  overall_news_tone: { label: "bearish|somewhat_bearish|neutral|somewhat_bullish|bullish", score: 0, rationale: "string" },
};

function pct(v: number | null): string {
  return v === null || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(2)}%`;
}

export function buildPrompt(input: MemoInput): string {
  const pc = input.priceContext;
  return [
    `You are a financial research assistant. Produce a NEWS TONE memo for ${input.ticker} (${input.companyName}) dated ${input.date}.`,
    `You are NOT an advisor. Never output buy, sell, hold, or price targets.`,
    ``,
    `Rules:`,
    `- Use ONLY the articles listed below. Do not invent facts or imply access to full article bodies.`,
    `- Every development MUST cite at least one article id (e.g. "a1") from the provided set in source_article_ids.`,
    `- Separate confirmed company events from analyst speculation.`,
    `- If evidence is thin, duplicated, or stale, say so in caveats, lower confidence, and return mostly-empty arrays.`,
    `- overall_news_tone reflects the tone of COVERAGE, not a stock forecast; rationale must reference the actual articles.`,
    ``,
    `Price context (factual; do NOT speculate on causation): latest close ${pc.latestClose ?? "n/a"} ${pc.currency ?? ""}; returns 1D ${pct(pc.returns.d1)}, 5D ${pct(pc.returns.d5)}, 1M ${pct(pc.returns.m1)}, 1Y ${pct(pc.returns.y1)}.`,
    ``,
    `Articles:`,
    ...input.articles.map((a) => `[${a.id}] (${a.source}, ${a.publishedAt}) ${a.headline}${a.summary ? ` — ${a.summary}` : ""}`),
    ``,
    `Respond with ONLY a JSON object (no markdown fences) matching this shape exactly:`,
    JSON.stringify(SHAPE_EXAMPLE, null, 2),
  ].join("\n");
}

export async function generateMemo(input: MemoInput): Promise<MemoOutput> {
  const ai = new GoogleGenAI({ apiKey: env.GEMINI_API_KEY! });
  let lastErr = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: env.GEMINI_MODEL,
        contents: buildPrompt(input),
        config: { responseMimeType: "application/json", temperature: 0.2 },
      });
      const parsed = memoOutputSchema.safeParse(JSON.parse(res.text ?? ""));
      if (parsed.success) {
        await recordSuccess("gemini", env.GEMINI_DAILY_LIMIT);
        return parsed.data;
      }
      lastErr = "schema validation failed";
    } catch (e) {
      lastErr = String(e);
    }
  }
  await recordError("gemini", env.GEMINI_DAILY_LIMIT, lastErr);
  throw new Error(`Gemini memo generation failed: ${lastErr}`);
}
```

- [ ] **Step 5: Run it to verify it passes**

Run: `pnpm test lib/providers/gemini.test.ts`
Expected: PASS (4 assertions). `generateMemo` is not unit-tested directly (network); it's mocked in Task 9.

- [ ] **Step 6: Commit**

```bash
git add package.json pnpm-lock.yaml lib/providers/gemini.ts lib/providers/gemini.test.ts
git commit -m "Add Gemini memo provider (schema, prompt, generate)"
```

---

## Task 9: memo-service

**Files:**
- Create: `lib/services/memo-service.ts`
- Test: `lib/services/memo-service.test.ts`

- [ ] **Step 1: Write the failing test**

Create `lib/services/memo-service.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  getTickerData, getCompanyByTicker, getNews,
  getRecentArticles, hasArticleNewerThan, getArticlesByIds,
  getMemoForDate, upsertMemo, canCall, generateMemo,
} = vi.hoisted(() => ({
  getTickerData: vi.fn(), getCompanyByTicker: vi.fn(), getNews: vi.fn(),
  getRecentArticles: vi.fn(), hasArticleNewerThan: vi.fn(), getArticlesByIds: vi.fn(),
  getMemoForDate: vi.fn(), upsertMemo: vi.fn(), canCall: vi.fn(), generateMemo: vi.fn(),
}));

vi.mock("@/lib/services/price-service", () => ({ getTickerData }));
vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker }));
vi.mock("@/lib/services/news-service", () => ({ getNews }));
vi.mock("@/lib/db/articles", () => ({ getRecentArticles, hasArticleNewerThan, getArticlesByIds }));
vi.mock("@/lib/db/daily-memos", () => ({ getMemoForDate, upsertMemo }));
vi.mock("@/lib/db/provider-state", () => ({ canCall }));
vi.mock("@/lib/providers/gemini", () => ({ generateMemo }));

import { getMemo } from "./memo-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", currency: "USD" };
const priceData = { bars: [{ close: 215.3 }], returns: { oneDay: -0.019, fiveDay: 0.02, oneMonth: 0.05, oneYear: 0.62 } };
const articleRow = (id: string) => ({ id, source: "finnhub", url: `https://ex.com/${id}`, title: `Title ${id}`, summary: "s", related: null, publishedAt: new Date("2026-05-24T12:00:00Z") });

const geminiOut = {
  ticker: "NVDA", date: "2026-05-25", one_sentence_takeaway: "Busy week.",
  bullish_developments: [
    { claim: "Real", why_it_matters: "x", source_article_ids: ["a1"], confidence: "medium" },
    { claim: "Hallucinated cite only", why_it_matters: "y", source_article_ids: ["a999"], confidence: "low" },
  ],
  bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: [], caveats: [],
  overall_news_tone: { label: "somewhat_bullish", score: 64, rationale: "Mostly positive." },
};

beforeEach(() => {
  [getTickerData, getCompanyByTicker, getNews, getRecentArticles, hasArticleNewerThan,
    getArticlesByIds, getMemoForDate, upsertMemo, canCall, generateMemo].forEach((m) => m.mockReset());
  getTickerData.mockResolvedValue(priceData);
  getCompanyByTicker.mockResolvedValue(company);
  getNews.mockResolvedValue({ articles: [], asOf: null });
  getRecentArticles.mockResolvedValue([articleRow("uuid-1")]);
  hasArticleNewerThan.mockResolvedValue(false);
  getArticlesByIds.mockImplementation(async (ids: string[]) => ids.map((id) => articleRow(id)));
  canCall.mockResolvedValue(true);
  generateMemo.mockResolvedValue(geminiOut);
  process.env.GEMINI_API_KEY = "test-key";
});

describe("getMemo", () => {
  it("serves a fresh cached memo without calling Gemini", async () => {
    getMemoForDate.mockResolvedValue({
      summaryJson: geminiOut, generatedAt: new Date(), model: "gemini-3.5-flash",
      basedOnArticleCount: 3, sourceArticleIds: ["uuid-1"],
    });
    const out = await getMemo("NVDA");
    expect(out.status).toBe("ok");
    expect(generateMemo).not.toHaveBeenCalled();
  });

  it("regenerates when a newer article exists", async () => {
    getMemoForDate.mockResolvedValue({
      summaryJson: geminiOut, generatedAt: new Date(Date.now() - 3600_000),
      model: "gemini-3.5-flash", basedOnArticleCount: 1, sourceArticleIds: ["uuid-1"],
    });
    hasArticleNewerThan.mockResolvedValue(true);
    const out = await getMemo("NVDA");
    expect(generateMemo).toHaveBeenCalled();
    expect(out.status).toBe("ok");
  });

  it("returns no_news when there are no recent articles", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    getRecentArticles.mockResolvedValue([]);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("no_news");
    expect(generateMemo).not.toHaveBeenCalled();
  });

  it("returns unavailable when the Gemini budget is exhausted", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    canCall.mockResolvedValue(false);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("unavailable");
  });

  it("drops developments citing unknown ids and persists the memo", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("ok");
    // 'a1' maps to the one input article; 'a999' is hallucinated → that development dropped
    expect(out.memo!.bullish_developments).toHaveLength(1);
    expect(upsertMemo).toHaveBeenCalledTimes(1);
  });

  it("returns error when Gemini throws", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    generateMemo.mockRejectedValue(new Error("gemini 500"));
    const out = await getMemo("NVDA");
    expect(out.status).toBe("error");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/services/memo-service.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `lib/services/memo-service.ts`:

```ts
import { env } from "@/lib/env";
import { getCompanyByTicker } from "@/lib/db/companies";
import { getTickerData } from "@/lib/services/price-service";
import { getNews } from "@/lib/services/news-service";
import { getRecentArticles, hasArticleNewerThan, getArticlesByIds, type ArticleRow } from "@/lib/db/articles";
import { getMemoForDate, upsertMemo } from "@/lib/db/daily-memos";
import { canCall } from "@/lib/db/provider-state";
import { generateMemo, type MemoInput, type MemoOutput } from "@/lib/providers/gemini";

export type MemoStatus = "ok" | "no_news" | "unavailable" | "error";
export type CitedArticle = { id: string; title: string; url: string; source: string };
export type MemoView = MemoOutput & { generatedAt: string; model: string; basedOnArticleCount: number };
export type MemoResult = { status: MemoStatus; memo: MemoView | null; citedArticles: CitedArticle[] };

const LOOKBACK_DAYS = 7;
const todayIso = () => new Date().toISOString().slice(0, 10);
function isoDaysAgo(n: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}

async function citedFrom(ids: string[]): Promise<CitedArticle[]> {
  const rows = await getArticlesByIds(ids);
  const byId = new Map(rows.map((r) => [r.id, r] as const));
  return ids
    .map((id) => byId.get(id))
    .filter((r): r is ArticleRow => !!r)
    .map((r) => ({ id: r.id, title: r.title, url: r.url, source: r.source }));
}

function view(memo: MemoOutput, generatedAt: Date, model: string, count: number): MemoView {
  return { ...memo, generatedAt: generatedAt.toISOString(), model, basedOnArticleCount: count };
}

export async function getMemo(ticker: string, opts: { force?: boolean } = {}): Promise<MemoResult> {
  const priceData = await getTickerData(ticker, "1y"); // ensures the company exists + price context
  const company = await getCompanyByTicker(ticker);
  if (!company) return { status: "no_news", memo: null, citedArticles: [] };
  const today = todayIso();

  const cached = await getMemoForDate(company.id, today);
  if (cached && !opts.force && !(await hasArticleNewerThan(company.id, cached.generatedAt))) {
    const memo = cached.summaryJson as MemoOutput;
    return {
      status: "ok",
      memo: view(memo, cached.generatedAt, cached.model, cached.basedOnArticleCount),
      citedArticles: await citedFrom(cached.sourceArticleIds as string[]),
    };
  }

  await getNews(ticker, { force: opts.force });
  const rows = await getRecentArticles(company.id, isoDaysAgo(LOOKBACK_DAYS));
  if (rows.length === 0) return { status: "no_news", memo: null, citedArticles: [] };

  if (!env.GEMINI_API_KEY || !(await canCall("gemini", env.GEMINI_DAILY_LIMIT))) {
    return { status: "unavailable", memo: null, citedArticles: [] };
  }

  const idMap = new Map<string, ArticleRow>();
  const input: MemoInput = {
    ticker: company.ticker,
    companyName: company.name,
    date: today,
    priceContext: {
      latestClose: priceData.bars.at(-1)?.close ?? null,
      currency: company.currency,
      returns: {
        d1: priceData.returns.oneDay, d5: priceData.returns.fiveDay,
        m1: priceData.returns.oneMonth, y1: priceData.returns.oneYear,
      },
    },
    articles: rows.map((r, i) => {
      const shortId = `a${i + 1}`;
      idMap.set(shortId, r);
      return { id: shortId, source: r.source, publishedAt: r.publishedAt.toISOString(), headline: r.title, summary: r.summary, related: r.related };
    }),
  };

  let out: MemoOutput;
  try {
    out = await generateMemo(input);
  } catch {
    return { status: "error", memo: null, citedArticles: [] };
  }

  // Citation guard: resolve short ids → UUIDs, drop hallucinated ids, drop zero-cite developments.
  const resolve = (ids: string[]) => ids.map((s) => idMap.get(s)?.id).filter((x): x is string => !!x);
  const fixGroup = (g: MemoOutput["bullish_developments"]) =>
    g.map((d) => ({ ...d, source_article_ids: resolve(d.source_article_ids) }))
     .filter((d) => d.source_article_ids.length > 0);

  const resolved: MemoOutput = {
    ...out,
    bullish_developments: fixGroup(out.bullish_developments),
    bearish_developments: fixGroup(out.bearish_developments),
    neutral_or_operational_updates: fixGroup(out.neutral_or_operational_updates),
  };

  const citedIds = [...new Set(
    [...resolved.bullish_developments, ...resolved.bearish_developments, ...resolved.neutral_or_operational_updates]
      .flatMap((d) => d.source_article_ids),
  )];

  await upsertMemo({
    companyId: company.id, memoDate: today, model: env.GEMINI_MODEL, summaryJson: resolved,
    toneLabel: resolved.overall_news_tone.label, toneScore: resolved.overall_news_tone.score,
    sourceArticleIds: citedIds, basedOnArticleCount: rows.length,
  });

  return {
    status: "ok",
    memo: view(resolved, new Date(), env.GEMINI_MODEL, rows.length),
    citedArticles: await citedFrom(citedIds),
  };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pnpm test lib/services/memo-service.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add lib/services/memo-service.ts lib/services/memo-service.test.ts
git commit -m "Add memo-service (cache-first generate, citation guard)"
```

---

## Task 10: /api/memo route

**Files:**
- Create: `app/api/memo/[symbol]/route.ts`

- [ ] **Step 1: Implement the route**

Create `app/api/memo/[symbol]/route.ts`:

```ts
import { type NextRequest } from "next/server";
import { z } from "zod";
import { getMemo } from "@/lib/services/memo-service";

export const dynamic = "force-dynamic";
export const maxDuration = 60; // news fetch + Gemini can be slow

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);

export async function GET(request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  const force = request.nextUrl.searchParams.get("force") === "1";
  try {
    return Response.json(await getMemo(t.data.toUpperCase(), { force }));
  } catch (e) {
    // Surface as a card-renderable error state rather than an HTTP failure.
    return Response.json({ status: "error", memo: null, citedArticles: [], detail: String(e) });
  }
}
```

- [ ] **Step 2: Verify typecheck + build**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add app/api/memo/
git commit -m "Add /api/memo route"
```

---

## Task 11: Tone util + ToneMeter

**Files:**
- Create: `lib/tone.ts`
- Test: `lib/tone.test.ts`
- Create: `components/tone-meter.tsx`

- [ ] **Step 1: Write the failing test**

Create `lib/tone.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { toneColor, toneLabelText } from "./tone";

describe("toneColor", () => {
  it("maps score ranges to colors", () => {
    expect(toneColor(10)).toBe(toneColor(0));   // bearish band
    expect(toneColor(50)).not.toBe(toneColor(10)); // neutral differs from bearish
    expect(toneColor(90)).not.toBe(toneColor(50)); // bullish differs from neutral
  });
});

describe("toneLabelText", () => {
  it("humanizes labels", () => {
    expect(toneLabelText("somewhat_bullish")).toBe("Somewhat bullish");
    expect(toneLabelText("neutral")).toBe("Neutral");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pnpm test lib/tone.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the util**

Create `lib/tone.ts`:

```ts
export type ToneLabel = "bearish" | "somewhat_bearish" | "neutral" | "somewhat_bullish" | "bullish";

export function toneLabelText(label: ToneLabel): string {
  const map: Record<ToneLabel, string> = {
    bearish: "Bearish",
    somewhat_bearish: "Somewhat bearish",
    neutral: "Neutral",
    somewhat_bullish: "Somewhat bullish",
    bullish: "Bullish",
  };
  return map[label];
}

export function toneColor(score: number): string {
  if (score <= 30) return "#ef4444"; // red
  if (score <= 45) return "#f59e0b"; // amber
  if (score <= 55) return "#a3a3a3"; // neutral grey
  if (score <= 70) return "#84cc16"; // lime
  return "#22c55e";                  // green
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pnpm test lib/tone.test.ts`
Expected: PASS.

- [ ] **Step 5: Implement the ToneMeter component**

Create `components/tone-meter.tsx`:

```tsx
import { toneColor, toneLabelText, type ToneLabel } from "@/lib/tone";

export function ToneMeter({ label, score }: { label: ToneLabel; score: number }) {
  const clamped = Math.max(0, Math.min(100, score));
  return (
    <div className="my-3">
      <div className="flex items-center justify-between text-xs text-neutral-400">
        <span>News Tone</span>
        <span>{score}/100 · {toneLabelText(label)}</span>
      </div>
      <div className="mt-1 h-2 w-full rounded bg-neutral-800">
        <div className="h-2 rounded" style={{ width: `${clamped}%`, background: toneColor(score) }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add lib/tone.ts lib/tone.test.ts components/tone-meter.tsx
git commit -m "Add tone util and ToneMeter component"
```

---

## Task 12: NewsTable + wire into the ticker page

**Files:**
- Create: `components/news-table.tsx`
- Modify: `app/ticker/[symbol]/page.tsx`

- [ ] **Step 1: Implement the NewsTable**

Create `components/news-table.tsx`:

```tsx
import type { ArticleRow } from "@/lib/db/articles";

function relTime(d: Date): string {
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

const sourceName = (s: string) => (s === "finnhub" ? "Finnhub" : "Yahoo");

export function NewsTable({ articles }: { articles: ArticleRow[] }) {
  if (articles.length === 0) {
    return <p className="text-sm text-neutral-500">No recent news.</p>;
  }
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="text-left text-xs uppercase text-neutral-500">
          <th className="py-1 pr-3 font-medium">Time</th>
          <th className="py-1 pr-3 font-medium">Source</th>
          <th className="py-1 font-medium">Headline</th>
        </tr>
      </thead>
      <tbody>
        {articles.map((a) => (
          <tr key={a.id} className="border-t border-neutral-800 align-top">
            <td className="whitespace-nowrap py-2 pr-3 text-neutral-400" title={a.publishedAt.toISOString()}>
              {relTime(a.publishedAt)}
            </td>
            <td className="whitespace-nowrap py-2 pr-3 text-neutral-400">{sourceName(a.source)}</td>
            <td className="py-2">
              <a href={a.url} target="_blank" rel="noopener noreferrer" className="text-neutral-100 hover:underline">
                {a.title}
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 2: Wire it into the page**

In `app/ticker/[symbol]/page.tsx`, add imports near the top:

```tsx
import { getNews } from "@/lib/services/news-service";
import { NewsTable } from "@/components/news-table";
```

After `const data = await getTickerData(symbol.toUpperCase(), "1y");`, add (sequential — the company must be ensured by `getTickerData` first):

```tsx
  const news = await getNews(data.ticker);
```

Then, inside the returned JSX, after the closing `</form>` and before `</main>`, add:

```tsx
      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Recent news</h2>
        <div className="mt-2"><NewsTable articles={news.articles} /></div>
      </section>
```

- [ ] **Step 3: Verify build**

Run: `pnpm build`
Expected: compiles successfully (the `/ticker/[symbol]` route builds).

- [ ] **Step 4: Commit**

```bash
git add components/news-table.tsx app/ticker/
git commit -m "Render recent news on the ticker page"
```

---

## Task 13: MemoCard + wire into the ticker page

**Files:**
- Create: `components/memo-card.tsx`
- Modify: `app/ticker/[symbol]/page.tsx`

- [ ] **Step 1: Implement the MemoCard**

Create `components/memo-card.tsx`:

```tsx
"use client";
import { useCallback, useEffect, useState } from "react";
import { ToneMeter } from "@/components/tone-meter";
import type { MemoResult, CitedArticle } from "@/lib/services/memo-service";

type Development = { claim: string; why_it_matters: string; source_article_ids: string[]; confidence: string };

function Chips({ ids, cited }: { ids: string[]; cited: CitedArticle[] }) {
  return (
    <>
      {ids.map((id) => {
        const idx = cited.findIndex((c) => c.id === id);
        const a = cited[idx];
        if (!a) return null;
        return (
          <a key={id} href={a.url} target="_blank" rel="noopener noreferrer"
            title={a.title}
            className="ml-1 rounded bg-neutral-800 px-1 text-xs text-sky-300 hover:bg-neutral-700">
            {idx + 1}
          </a>
        );
      })}
    </>
  );
}

function Group({ title, items, cited }: { title: string; items: Development[]; cited: CitedArticle[] }) {
  if (items.length === 0) return null;
  return (
    <div className="mt-4">
      <h3 className="text-xs font-medium uppercase text-neutral-500">{title}</h3>
      <ul className="mt-1 space-y-2">
        {items.map((d, i) => (
          <li key={i} className="text-sm">
            <span className="text-neutral-100">{d.claim}</span>
            <span className="text-neutral-500"> — {d.why_it_matters}</span>
            <span className="ml-1 text-xs text-neutral-600">({d.confidence})</span>
            <Chips ids={d.source_article_ids} cited={cited} />
          </li>
        ))}
      </ul>
    </div>
  );
}

export function MemoCard({ symbol }: { symbol: string }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<MemoResult | null>(null);

  const load = useCallback(async (force: boolean) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/memo/${symbol}${force ? "?force=1" : ""}`);
      setData(await res.json());
    } catch {
      setData({ status: "error", memo: null, citedArticles: [] });
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => { void load(false); }, [load]);

  if (loading) return <p className="text-sm text-neutral-500">Generating today&apos;s memo…</p>;
  if (!data || data.status === "error")
    return (
      <div className="text-sm text-neutral-500">
        Couldn&apos;t generate the memo. <button onClick={() => load(true)} className="underline">Try again</button>
      </div>
    );
  if (data.status === "no_news") return <p className="text-sm text-neutral-500">No recent news in the last 7 days.</p>;
  if (data.status === "unavailable")
    return <p className="text-sm text-neutral-500">Memo unavailable — Gemini key missing or daily limit reached.</p>;

  const m = data.memo!;
  return (
    <div className="rounded-lg ring-1 ring-neutral-800 p-4">
      <p className="text-base text-neutral-100">{m.one_sentence_takeaway}</p>
      <ToneMeter label={m.overall_news_tone.label} score={m.overall_news_tone.score} />
      <Group title="Bullish" items={m.bullish_developments} cited={data.citedArticles} />
      <Group title="Bearish" items={m.bearish_developments} cited={data.citedArticles} />
      <Group title="Neutral / operational" items={m.neutral_or_operational_updates} cited={data.citedArticles} />
      {m.watch_items.length > 0 && (
        <div className="mt-4">
          <h3 className="text-xs font-medium uppercase text-neutral-500">Watch items</h3>
          <ul className="mt-1 list-disc pl-5 text-sm text-neutral-300">{m.watch_items.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}
      {m.caveats.length > 0 && (
        <div className="mt-4">
          <h3 className="text-xs font-medium uppercase text-neutral-500">Caveats</h3>
          <ul className="mt-1 list-disc pl-5 text-sm text-neutral-400">{m.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </div>
      )}
      <div className="mt-4 flex items-center justify-between border-t border-neutral-800 pt-2 text-xs text-neutral-600">
        <span>
          Generated by {m.model} from {m.basedOnArticleCount} sources · every claim links to its source · News Tone reflects coverage tone, not a forecast · not investment advice.
        </span>
        <button onClick={() => load(true)} className="ml-3 shrink-0 underline">Regenerate</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire it into the page**

In `app/ticker/[symbol]/page.tsx`, add the import:

```tsx
import { MemoCard } from "@/components/memo-card";
```

Insert this section **before** the "Recent news" section added in Task 12 (so order is: header + chart → memo → news):

```tsx
      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Daily memo</h2>
        <p className="mb-2 text-xs text-neutral-600">Research assistant, not investment advice.</p>
        <MemoCard symbol={data.ticker} />
      </section>
```

- [ ] **Step 3: Verify build**

Run: `pnpm build`
Expected: compiles successfully.

- [ ] **Step 4: Commit**

```bash
git add components/memo-card.tsx app/ticker/
git commit -m "Render the daily memo card on the ticker page"
```

---

## Task 14: E2E smoke + full verification

**Files:**
- Modify: `tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Add the memo/news smoke test**

Append to `tests/e2e/smoke.spec.ts`:

```ts
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
  await expect(page.getByText(/News Tone/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent news" })).toBeVisible();
});
```

(The news table is server-rendered, so it isn't interceptable; this asserts the section heading renders. Live news rows depend on real provider data and aren't asserted in the smoke run.)

- [ ] **Step 2: Run the E2E suite**

Run: `pnpm test:e2e`
Expected: all tests pass (the 2 existing + this new one). The dev server must be able to start; `getMemo` is never hit live because the route is intercepted.

- [ ] **Step 3: Run the full unit/integration suite + typecheck**

Run: `pnpm test && pnpm exec tsc --noEmit`
Expected: all unit tests pass; no type errors.

- [ ] **Step 4: Manual browser verification (real keys)**

Run: `pnpm dev`, then open `http://localhost:3000/ticker/NVDA`. Confirm:
- Prices + chart paint immediately.
- The "Recent news" table populates from Finnhub/Yahoo within the last 7 days (duplicates collapsed).
- The memo card shows a spinner, then a takeaway + tone meter + grouped developments whose numbered chips link to source articles.
- The transparency footer is present; no buy/sell/hold text appears.
- Clicking **Regenerate** re-runs generation. Reloading the page serves the cached memo (no spinner delay beyond the fetch).
- Try a ticker with no news and a stock with sparse coverage to see `no_news` / caveats behavior.

If the Gemini structured-output call errors at runtime, confirm the `@google/genai` call shape in `lib/providers/gemini.ts` against the installed SDK version (the `config` field names can differ by version) and adjust.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/smoke.spec.ts
git commit -m "Add SP2 news/memo E2E smoke test"
```

- [ ] **Step 6: Final review**

Dispatch a final code review over the whole SP2 diff (all commits since the SP2 spec commit `4d32a76`). Confirm: degrade-never-crash holds on every provider/Gemini failure path; the citation guard prevents unsourced claims; no secrets logged; no AI-authorship traces in code or commits; acceptance criteria in spec §7.10 are met.

---

## Self-Review (completed during planning)

**Spec coverage (§7):** §7.2 tables → T2; env → T1; `NewsArticle` → T3; Finnhub → T3; Yahoo RSS → T4; dedupe → T5; repos → T6; `news-service` → T7; Gemini contract + guards → T8 (contract) + T9 (code-enforced guards); `memo-service` cache-first/staleness/no_news/unavailable → T9; `/api/memo` + maxDuration → T10; tone meter → T11; news table → T12; memo card + transparency footer → T13; error handling → spread across T3/T4/T7/T9/T10/T13; testing → unit (T3–T9, T11), E2E (T14); acceptance criteria → T14 Step 4. No gaps.

**Placeholder scan:** none — every code step contains complete, paste-ready code; commands have expected output.

**Type consistency:** `NewsArticle` (T3) is consumed by parsers (T3/T4), dedupe (T5), articles repo (T6). `ArticleRow` (T6) flows to `news-service` (T7), `memo-service` (T9), `NewsTable` (T12). `MemoOutput`/`MemoInput` (T8) are used by `memo-service` (T9). `MemoResult`/`CitedArticle`/`MemoView` (T9) are consumed by the route (T10) and `MemoCard` (T13). `ToneLabel` (T11) is used by `ToneMeter` (T11) and indirectly via `overall_news_tone.label`. `getMemoForDate`/`upsertMemo` (T6) match their `memo-service` calls (T9). Consistent.
