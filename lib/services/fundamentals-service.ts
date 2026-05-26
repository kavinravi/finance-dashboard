import { getCompanyByTicker, setCik } from "@/lib/db/companies";
import { getTickerData } from "@/lib/services/price-service";
import { getFundamentalsSnapshot, upsertFundamentalsSnapshot, type FundamentalsRow } from "@/lib/db/fundamentals";
import { sec } from "@/lib/providers/sec";
import { extractConcepts } from "@/lib/fundamentals/extract";
import { deriveMetrics } from "@/lib/fundamentals/derive";
import type { FundamentalConcepts, FundamentalsMeta, FundamentalsView } from "@/lib/types";

export type FundamentalsStatus = "ok" | "not_applicable" | "unavailable" | "error";
export type FundamentalsAsOf = FundamentalsMeta & { edgarUrl: string | null };
export type FundamentalsResult = {
  status: FundamentalsStatus;
  view: FundamentalsView | null;
  asOf: FundamentalsAsOf | null;
  source: "sec_edgar" | null;
};

const TTL_MS = 7 * 24 * 60 * 60 * 1000;
const edgarUrl = (cik: string) =>
  `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&type=10-K`;

const metaFromSnap = (s: FundamentalsRow): FundamentalsMeta => ({
  fiscalYear: s.fiscalYear,
  incomePeriodEnd: s.incomePeriodEnd,
  balanceSheetAsOf: s.balanceSheetAsOf,
  filingForm: s.filingForm,
  filedAt: s.filedAt,
});

function ok(concepts: FundamentalConcepts, meta: FundamentalsMeta, cik: string | null, latestClose: number | null): FundamentalsResult {
  return {
    status: "ok",
    view: deriveMetrics(concepts, latestClose),
    asOf: { ...meta, edgarUrl: cik ? edgarUrl(cik) : null },
    source: "sec_edgar",
  };
}

export async function getFundamentals(ticker: string): Promise<FundamentalsResult> {
  // Ensures the company exists + provides the latest cached close (no extra provider call when bars are fresh).
  const priceData = await getTickerData(ticker, "1y");
  const company = await getCompanyByTicker(ticker);
  if (!company) return { status: "unavailable", view: null, asOf: null, source: null };
  if (company.assetType !== "stock") return { status: "not_applicable", view: null, asOf: null, source: null };

  const latestClose = priceData.bars.at(-1)?.close ?? null;
  const snap = await getFundamentalsSnapshot(company.id);

  // Cache-first.
  if (snap && snap.expiresAt.getTime() > Date.now()) {
    return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), company.cik, latestClose);
  }

  // Ensure CIK.
  let cik = company.cik;
  if (!cik) {
    cik = await sec.resolveCik(company.ticker);
    if (cik) await setCik(company.id, cik);
  }
  if (!cik) {
    if (snap) return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), null, latestClose);
    return { status: "unavailable", view: null, asOf: null, source: null };
  }

  // Fetch + extract + cache.
  const raw = await sec.fetchCompanyFacts(cik);
  if (!raw) {
    if (snap) return ok(snap.conceptsJson as FundamentalConcepts, metaFromSnap(snap), cik, latestClose);
    return { status: "error", view: null, asOf: null, source: null };
  }
  const { concepts, meta } = extractConcepts(raw);
  await upsertFundamentalsSnapshot({
    companyId: company.id, conceptsJson: concepts, meta, source: "sec_edgar",
    expiresAt: new Date(Date.now() + TTL_MS),
  });
  return ok(concepts, meta, cik, latestClose);
}
