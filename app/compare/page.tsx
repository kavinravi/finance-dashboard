import Link from "next/link";
import { compareTickers } from "@/lib/services/comparison-service";
import { ComparisonChart } from "@/components/comparison-chart";
import { formatPercent } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function ComparePage({
  searchParams,
}: { searchParams: Promise<{ primary?: string; comparison?: string; range?: string }> }) {
  const sp = await searchParams;
  const primary = (sp.primary ?? "").toUpperCase();
  const comparison = (sp.comparison ?? "").toUpperCase();

  if (!primary || !comparison) {
    return <main className="mx-auto max-w-4xl px-4 pt-16"><p>Provide both a primary and comparison ticker.</p></main>;
  }

  const r = await compareTickers(primary, comparison, "1y");
  const noOverlap = r.dates.length === 0;

  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href={`/ticker/${primary}`} className="text-sm text-neutral-500">← {primary}</Link>
      <h1 className="mt-4 text-2xl font-semibold">
        <span className="font-mono">{primary}</span> vs <span className="font-mono">{comparison}</span>
      </h1>

      {noOverlap ? (
        <p className="mt-6 text-neutral-400">No overlapping price history to compare.</p>
      ) : (
        <>
          <p className="mt-2 text-sm text-neutral-400">
            Relative return (1Y window): <span className={r.relativeReturn! >= 0 ? "text-emerald-400" : "text-red-400"}>
              {formatPercent(r.relativeReturn)}</span>
          </p>
          <div className="mt-6">
            <ComparisonChart dates={r.dates} primaryTicker={primary} comparisonTicker={comparison}
              primary={r.primary.normalized} comparison={r.comparison.normalized} />
          </div>
          <table className="mt-8 w-full text-sm">
            <thead><tr className="text-left text-neutral-500">
              <th className="py-1">Metric</th><th>{primary}</th><th>{comparison}</th></tr></thead>
            <tbody>
              <tr><td className="py-1 text-neutral-400">5D volatility</td>
                <td>{formatPercent(r.primary.volatility)}</td><td>{formatPercent(r.comparison.volatility)}</td></tr>
              <tr><td className="py-1 text-neutral-400">Max drawdown</td>
                <td className="text-red-400">{formatPercent(r.primary.maxDrawdown)}</td>
                <td className="text-red-400">{formatPercent(r.comparison.maxDrawdown)}</td></tr>
            </tbody>
          </table>
        </>
      )}
    </main>
  );
}
