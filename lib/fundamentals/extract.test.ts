import { describe, it, expect } from "vitest";
import { extractConcepts } from "./extract";
import type { RawCompanyFacts } from "@/lib/types";

function usd(facts: object[]) { return { units: { USD: facts } }; }

const RAW: RawCompanyFacts = {
  cik: 320193,
  entityName: "Apple Inc.",
  facts: {
    "us-gaap": {
      // No "Revenues" tag -> must fall back to RevenueFromContractWithCustomerExcludingAssessedTax
      RevenueFromContractWithCustomerExcludingAssessedTax: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 391035000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
        { start: "2024-09-29", end: "2024-12-28", val: 124300000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }, // quarter — ignored for FY flow
        { start: "2022-09-25", end: "2023-09-30", val: 383285000000, fy: 2023, fp: "FY", form: "10-K", filed: "2023-11-03" }, // older FY
      ]),
      NetIncomeLoss: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 93736000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      EarningsPerShareDiluted: { units: { "USD/shares": [
        { start: "2023-10-01", end: "2024-09-28", val: 6.08, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ] } },
      OperatingIncomeLoss: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 123216000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      GrossProfit: usd([
        { start: "2023-10-01", end: "2024-09-28", val: 180683000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
      ]),
      Assets: usd([
        { end: "2024-09-28", val: 364980000000, fy: 2024, fp: "FY", form: "10-K", filed: "2024-11-01" },
        { end: "2024-12-28", val: 344085000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }, // newer instant — wins
      ]),
      Liabilities: usd([{ end: "2024-12-28", val: 277327000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      StockholdersEquity: usd([{ end: "2024-12-28", val: 66758000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      AssetsCurrent: usd([{ end: "2024-12-28", val: 133240000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
      LiabilitiesCurrent: usd([{ end: "2024-12-28", val: 144365000000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" }]),
    },
    dei: {
      EntityCommonStockSharesOutstanding: { units: { shares: [
        { end: "2025-01-17", val: 15022073000, fy: 2025, fp: "Q1", form: "10-Q", filed: "2025-01-31" },
      ] } },
    },
  },
};

describe("extractConcepts", () => {
  const { concepts, meta } = extractConcepts(RAW);

  it("picks the latest FY value for flow metrics (via tag fallback)", () => {
    expect(concepts.revenue).toBe(391035000000);   // 2024 FY, not the older FY or the quarter
    expect(concepts.netIncome).toBe(93736000000);
    expect(concepts.eps).toBe(6.08);
    expect(concepts.operatingIncome).toBe(123216000000);
    expect(concepts.grossProfit).toBe(180683000000);
  });

  it("picks the latest instant for balance-sheet metrics + shares", () => {
    expect(concepts.assets).toBe(344085000000);     // newer Q1 instant beats the FY instant
    expect(concepts.liabilities).toBe(277327000000);
    expect(concepts.equity).toBe(66758000000);
    expect(concepts.currentAssets).toBe(133240000000);
    expect(concepts.currentLiabilities).toBe(144365000000);
    expect(concepts.sharesOutstanding).toBe(15022073000);
  });

  it("records provenance from the income-statement anchor + the latest balance sheet", () => {
    expect(meta.fiscalYear).toBe(2024);
    expect(meta.incomePeriodEnd).toBe("2024-09-28");
    expect(meta.filingForm).toBe("10-K");
    expect(meta.filedAt).toBe("2024-11-01");
    expect(meta.balanceSheetAsOf).toBe("2024-12-28");
  });

  it("returns nulls for missing tags and an all-null result for empty input", () => {
    const partial = extractConcepts({ facts: { "us-gaap": {} } });
    expect(partial.concepts.revenue).toBeNull();
    expect(partial.meta.fiscalYear).toBeNull();
    const none = extractConcepts(null);
    expect(none.concepts.netIncome).toBeNull();
  });
});
