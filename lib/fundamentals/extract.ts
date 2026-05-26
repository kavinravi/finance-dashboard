import type { RawCompanyFacts, SecFact, ExtractedFundamentals } from "@/lib/types";

function durationDays(f: SecFact): number {
  if (!f.start) return 0;
  return (Date.parse(f.end) - Date.parse(f.start)) / 86400000;
}

// Most-recent full fiscal-year fact (fp:"FY", ~annual duration). Tie-break by latest filing.
function pickAnnualFlow(facts: SecFact[]): SecFact | null {
  const annual = facts.filter((f) => f.fp === "FY" && durationDays(f) >= 300);
  if (annual.length === 0) return null;
  return [...annual].sort(
    (a, b) => b.end.localeCompare(a.end) || (b.filed ?? "").localeCompare(a.filed ?? ""),
  )[0];
}

// Latest instant (balance-sheet / shares). Tie-break by latest filing.
function pickLatestInstant(facts: SecFact[]): SecFact | null {
  if (facts.length === 0) return null;
  return [...facts].sort(
    (a, b) => b.end.localeCompare(a.end) || (b.filed ?? "").localeCompare(a.filed ?? ""),
  )[0];
}

// First non-empty fact array among the candidate tags, in the given unit bucket.
function factsFor(raw: RawCompanyFacts, taxonomy: string, tags: string[], unit: string): SecFact[] {
  const tax = raw.facts?.[taxonomy];
  if (!tax) return [];
  for (const tag of tags) {
    const arr = tax[tag]?.units?.[unit];
    if (Array.isArray(arr) && arr.length > 0) return arr;
  }
  return [];
}

const EMPTY: ExtractedFundamentals = {
  concepts: {
    revenue: null, netIncome: null, eps: null, operatingIncome: null, grossProfit: null,
    assets: null, liabilities: null, equity: null, currentAssets: null, currentLiabilities: null,
    sharesOutstanding: null,
  },
  meta: { fiscalYear: null, incomePeriodEnd: null, balanceSheetAsOf: null, filingForm: null, filedAt: null },
};

export function extractConcepts(raw: RawCompanyFacts | null): ExtractedFundamentals {
  if (!raw || !raw.facts) return EMPTY;
  const gaap = (tags: string[], unit = "USD") => factsFor(raw, "us-gaap", tags, unit);

  const revenue = pickAnnualFlow(gaap(["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]));
  const netIncome = pickAnnualFlow(gaap(["NetIncomeLoss"]));
  const eps = pickAnnualFlow(gaap(["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares"));
  const operatingIncome = pickAnnualFlow(gaap(["OperatingIncomeLoss"]));
  const grossProfit = pickAnnualFlow(gaap(["GrossProfit"]));
  const assets = pickLatestInstant(gaap(["Assets"]));
  const liabilities = pickLatestInstant(gaap(["Liabilities"]));
  const equity = pickLatestInstant(gaap(["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]));
  const currentAssets = pickLatestInstant(gaap(["AssetsCurrent"]));
  const currentLiabilities = pickLatestInstant(gaap(["LiabilitiesCurrent"]));
  const shares = pickLatestInstant(factsFor(raw, "dei", ["EntityCommonStockSharesOutstanding"], "shares"));

  const anchor = netIncome ?? revenue;

  return {
    concepts: {
      revenue: revenue?.val ?? null,
      netIncome: netIncome?.val ?? null,
      eps: eps?.val ?? null,
      operatingIncome: operatingIncome?.val ?? null,
      grossProfit: grossProfit?.val ?? null,
      assets: assets?.val ?? null,
      liabilities: liabilities?.val ?? null,
      equity: equity?.val ?? null,
      currentAssets: currentAssets?.val ?? null,
      currentLiabilities: currentLiabilities?.val ?? null,
      sharesOutstanding: shares?.val ?? null,
    },
    meta: {
      fiscalYear: anchor?.fy ?? null,
      incomePeriodEnd: anchor?.end ?? null,
      balanceSheetAsOf: assets?.end ?? equity?.end ?? null,
      filingForm: anchor?.form ?? null,
      filedAt: anchor?.filed ?? null,
    },
  };
}
