import type { PriceBar, SearchResult, CompanyProfile } from "@/lib/types";

export interface MarketDataProvider {
  name: string;
  search(query: string): Promise<SearchResult[]>;
  profile(ticker: string): Promise<CompanyProfile | null>;
  dailyPrices(ticker: string, from: string, to: string): Promise<PriceBar[]>;
}
export type { PriceBar, SearchResult, CompanyProfile };
