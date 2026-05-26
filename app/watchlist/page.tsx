import { getWatchlistOverlay } from "@/lib/services/watchlist-service";
import { OverlayChart } from "@/components/overlay-chart";
import { WatchlistManager } from "@/components/watchlist-manager";

export const dynamic = "force-dynamic";

export default async function WatchlistPage() {
  const { tickers, overlay } = await getWatchlistOverlay();
  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <h1 className="text-2xl font-semibold">Watchlist</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Normalized to 100 at the start of the common window. Research only — not advice.
      </p>
      <div className="mt-4"><WatchlistManager initialTickers={tickers} /></div>
      {overlay.series.length > 0 ? (
        <div className="mt-6"><OverlayChart dates={overlay.dates} series={overlay.series} /></div>
      ) : (
        <p className="mt-6 text-neutral-400">Add tickers above to see them overlaid here.</p>
      )}
    </main>
  );
}
