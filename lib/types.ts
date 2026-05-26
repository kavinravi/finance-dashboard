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

// Display-ready metrics (null -> "N/A"). Margins/ROE/ROA are fractions (x100 in UI).
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
