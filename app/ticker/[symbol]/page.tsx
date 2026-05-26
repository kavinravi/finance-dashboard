import { getTickerData } from "@/lib/services/price-service";
import { ReturnsTable } from "@/components/returns-table";
import { TickerCharts } from "@/components/ticker-charts";
import { StalenessBadge } from "@/components/staleness-badge";
import { FundamentalsCard } from "@/components/fundamentals-card";
import { formatPrice } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function TickerChartsPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const data = await getTickerData(symbol.toUpperCase(), "1y");
  const latest = data.bars.at(-1)?.close ?? null;

  if (data.bars.length === 0) {
    return (
      <p className="mt-8">
        Couldn&apos;t resolve <span className="font-mono">{symbol.toUpperCase()}</span>. Try another ticker.
      </p>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-baseline justify-between">
        <div className="text-2xl">{formatPrice(latest)}</div>
        <StalenessBadge stale={data.stale} lastBarDate={data.lastBarDate} source={data.source} />
      </div>

      <div className="mt-4"><ReturnsTable returns={data.returns} /></div>

      <div className="mt-6">
        <TickerCharts
          bars={data.bars}
          indicators={{
            ma20: data.indicators.ma20, ma50: data.indicators.ma50, rsi14: data.indicators.rsi14,
            macdLine: data.indicators.macdLine, macdSignal: data.indicators.macdSignal, macdHistogram: data.indicators.macdHistogram,
          }}
        />
      </div>

      <form action="/compare" className="mt-8 flex items-center gap-2">
        <input type="hidden" name="primary" value={data.ticker} />
        <label className="text-sm text-neutral-400">Compare against</label>
        <input name="comparison" placeholder="e.g. SPY"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900">Go</button>
      </form>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Fundamentals</h2>
        <p className="mb-2 text-xs text-neutral-600">From official SEC filings.</p>
        <FundamentalsCard symbol={data.ticker} />
      </section>
    </div>
  );
}
