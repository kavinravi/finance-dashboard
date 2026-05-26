import {
  pgTable, uuid, text, date, timestamp, doublePrecision, bigint, integer, unique, jsonb,
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
  cik: text("cik"), // SEC CIK, 10-digit zero-padded; resolved once, reused
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

export const providerState = pgTable("provider_state", {
  provider: text("provider").primaryKey(),       // "fmp" | "yahoo"
  callsToday: integer("calls_today").notNull().default(0),
  dailyLimit: integer("daily_limit").notNull(),
  resetAt: timestamp("reset_at", { withTimezone: true }).notNull(),
  lastSuccessAt: timestamp("last_success_at", { withTimezone: true }),
  lastErrorAt: timestamp("last_error_at", { withTimezone: true }),
  lastError: text("last_error"),
});

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
