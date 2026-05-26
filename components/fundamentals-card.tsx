"use client";
import { useCallback, useEffect, useState } from "react";
import type { FundamentalsResult } from "@/lib/services/fundamentals-service";
import { formatLargeCurrency, formatMultiple, formatRatioPercent } from "@/lib/formatters";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-0.5 text-sm">
      <span className="text-neutral-500">{label}</span>
      <span className="font-mono text-neutral-100">{value}</span>
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <h3 className="text-xs font-medium uppercase text-neutral-500">{title}</h3>
      <div className="mt-1">{children}</div>
    </div>
  );
}

export function FundamentalsCard({ symbol }: { symbol: string }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<FundamentalsResult | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/fundamentals/${symbol}`);
      setData(await res.json());
    } catch {
      setData({ status: "error", view: null, asOf: null, source: null });
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => { void load(); }, [load]);

  if (loading) return <p className="text-sm text-neutral-500">Loading fundamentals…</p>;
  if (!data || data.status === "error")
    return (
      <div className="text-sm text-neutral-500">
        Couldn&apos;t load fundamentals. <button onClick={() => load()} className="underline">Try again</button>
      </div>
    );
  if (data.status === "not_applicable")
    return <p className="text-sm text-neutral-500">Fundamentals aren&apos;t available for ETFs/indexes.</p>;
  if (data.status === "unavailable" || !data.view)
    return <p className="text-sm text-neutral-500">Fundamentals unavailable — SEC data couldn&apos;t be fetched.</p>;

  const v = data.view;
  const a = data.asOf;
  return (
    <div className="rounded-lg ring-1 ring-neutral-800 p-4">
      <div className="grid grid-cols-1 gap-x-8 sm:grid-cols-2">
        <Group title="Valuation">
          <Row label="Market Cap" value={formatLargeCurrency(v.marketCap)} />
          <Row label="P/E (FY EPS)" value={formatMultiple(v.peRatio)} />
          <Row label="P/S" value={formatMultiple(v.psRatio)} />
        </Group>
        <Group title="Profitability">
          <Row label="Gross Margin" value={formatRatioPercent(v.grossMargin)} />
          <Row label="ROE" value={formatRatioPercent(v.roe)} />
          <Row label="ROA" value={formatRatioPercent(v.roa)} />
          <Row label="Operating Income" value={formatLargeCurrency(v.operatingIncome)} />
        </Group>
        <Group title="Financial Health">
          <Row label="Current Ratio" value={formatMultiple(v.currentRatio)} />
          <Row label="Debt/Equity" value={formatMultiple(v.debtToEquity)} />
          <Row label="Assets" value={formatLargeCurrency(v.assets)} />
          <Row label="Liabilities" value={formatLargeCurrency(v.liabilities)} />
          <Row label="Equity" value={formatLargeCurrency(v.equity)} />
        </Group>
        <Group title="Latest Financials">
          <Row label="Revenue" value={formatLargeCurrency(v.revenue)} />
          <Row label="Net Income" value={formatLargeCurrency(v.netIncome)} />
          <Row label="EPS (diluted)" value={v.eps === null ? "N/A" : `$${v.eps.toFixed(2)}`} />
        </Group>
      </div>
      <div className="mt-4 border-t border-neutral-800 pt-2 text-xs text-neutral-600">
        Source: SEC EDGAR
        {a?.fiscalYear ? ` · FY${a.fiscalYear} ${a.filingForm ?? ""}${a.filedAt ? ` filed ${a.filedAt}` : ""}` : ""}
        {a?.balanceSheetAsOf ? ` · balance sheet as of ${a.balanceSheetAsOf}` : ""}
        {a?.edgarUrl ? <> · <a href={a.edgarUrl} target="_blank" rel="noopener noreferrer" className="underline">filings on SEC.gov</a></> : null}
      </div>
    </div>
  );
}
