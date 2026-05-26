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
