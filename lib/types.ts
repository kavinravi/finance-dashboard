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
