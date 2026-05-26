import Link from "next/link";
import { getTickerData } from "@/lib/services/price-service";
import { getNews } from "@/lib/services/news-service";
import { ReturnsTable } from "@/components/returns-table";
import { PriceChart } from "@/components/price-chart";
import { StalenessBadge } from "@/components/staleness-badge";
import { NewsTable } from "@/components/news-table";
import { MemoCard } from "@/components/memo-card";
import { FundamentalsCard } from "@/components/fundamentals-card";
import { formatPrice } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function TickerPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const data = await getTickerData(symbol.toUpperCase(), "1y");
  const latest = data.bars.at(-1)?.close ?? null;

  if (data.bars.length === 0) {
    return (
      <main className="mx-auto max-w-4xl px-4 pt-16">
        <Link href="/" className="text-sm text-neutral-500">← Search</Link>
        <p className="mt-8">Couldn&apos;t resolve <span className="font-mono">{symbol.toUpperCase()}</span>. Try another ticker.</p>
      </main>
    );
  }

  const news = await getNews(data.ticker);

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href="/" className="text-sm text-neutral-500">← Search</Link>
      <div className="mt-4 flex items-baseline justify-between">
        <div>
          <h1 className="font-mono text-3xl font-semibold">{data.ticker}</h1>
          <div className="text-2xl">{formatPrice(latest)}</div>
        </div>
        <StalenessBadge stale={data.stale} lastBarDate={data.lastBarDate} source={data.source} />
      </div>

      <div className="mt-4"><ReturnsTable returns={data.returns} /></div>
      <div className="mt-6"><PriceChart bars={data.bars} ma20={data.indicators.ma20} ma50={data.indicators.ma50} /></div>

      <form action="/compare" className="mt-8 flex items-center gap-2">
        <input type="hidden" name="primary" value={data.ticker} />
        <label className="text-sm text-neutral-400">Compare against</label>
        <input name="comparison" placeholder="e.g. SPY"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900">Go</button>
      </form>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Daily memo</h2>
        <p className="mb-2 text-xs text-neutral-600">Research assistant, not investment advice.</p>
        <MemoCard symbol={data.ticker} />
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Fundamentals</h2>
        <p className="mb-2 text-xs text-neutral-600">From official SEC filings.</p>
        <FundamentalsCard symbol={data.ticker} />
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Recent news</h2>
        <div className="mt-2"><NewsTable articles={news.articles} /></div>
      </section>
    </main>
  );
}
