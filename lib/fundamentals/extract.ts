import type { RawCompanyFacts, SecFact, ExtractedFundamentals } from "@/lib/types";

function durationDays(f: SecFact): number {
  if (!f.start) return 0;
  return (Date.parse(f.end) - Date.parse(f.start)) / 86400000;
}

// A fact tagged with its candidate-tag priority (0 = most-preferred tag).
type RankedFact = SecFact & { __pri: number };

// Latest period first; for the same period prefer the earlier (preferred) candidate tag;
// then prefer the latest filing.
function byRecency(a: RankedFact, b: RankedFact): number {
  return b.end.localeCompare(a.end) || a.__pri - b.__pri || (b.filed ?? "").localeCompare(a.filed ?? "");
}

// Most-recent full fiscal-year fact (fp:"FY", ~annual duration).
function pickAnnualFlow(facts: RankedFact[]): RankedFact | null {
  const annual = facts.filter((f) => f.fp === "FY" && durationDays(f) >= 300);
  if (annual.length === 0) return null;
  return [...annual].sort(byRecency)[0];
}

// Latest instant (balance-sheet / shares).
function pickLatestInstant(facts: RankedFact[]): RankedFact | null {
  if (facts.length === 0) return null;
  return [...facts].sort(byRecency)[0];
}

// Merge facts from ALL candidate tags (in the given unit bucket), tagging each with its
// candidate priority. Merging (rather than first-non-empty) handles XBRL tag migration:
// e.g. a company that reported revenue under `Revenues` until 2018, then switched to
// `RevenueFromContractWithCustomerExcludingAssessedTax` — the stale tag still carries an
// old value, so first-non-empty would return years-stale data. Picking the latest period
// across the union fixes that; the priority tiebreak keeps a preferred tag (e.g. diluted
// EPS over basic) when multiple tags report the same period.
function collectFacts(raw: RawCompanyFacts, taxonomy: string, tags: string[], unit: string): RankedFact[] {
  const tax = raw.facts?.[taxonomy];
  if (!tax) return [];
  const out: RankedFact[] = [];
  tags.forEach((tag, pri) => {
    const arr = tax[tag]?.units?.[unit];
    if (Array.isArray(arr)) for (const f of arr) out.push({ ...f, __pri: pri });
  });
  return out;
}

const EMPTY: ExtractedFundamentals = {
  concepts: {
    revenue: null, netIncome: null, eps: null, operatingIncome: null, grossProfit: null,
    assets: null, liabilities: null, equity: null, currentAssets: null, currentLiabilities: null,
    sharesOutstanding: null,
  },
  meta: { fiscalYear: null, incomePeriodEnd: null, balanceSheetAsOf: null, filingForm: null, filedAt: null },
};

// Reads only the `us-gaap` and `dei` taxonomies. IFRS filers (many foreign 20-F issuers,
// taxonomy `ifrs-full`) and filers whose annuals lack an `fp:"FY"` marker yield all-null —
// the card then degrades to "unavailable" rather than showing wrong numbers. Deliberate
// MVP scope, not a bug.
export function extractConcepts(raw: RawCompanyFacts | null): ExtractedFundamentals {
  if (!raw || !raw.facts) return EMPTY;
  const gaap = (tags: string[], unit = "USD") => collectFacts(raw, "us-gaap", tags, unit);

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
  const shares = pickLatestInstant(collectFacts(raw, "dei", ["EntityCommonStockSharesOutstanding"], "shares"));

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
