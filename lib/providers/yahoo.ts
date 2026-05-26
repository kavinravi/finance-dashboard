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
    return null;
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
