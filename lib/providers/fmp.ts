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
    assetType: "stock" as AssetType,
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
