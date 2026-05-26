import { getWatchlist } from "@/lib/db/watchlist";
import { getTickerData } from "@/lib/services/price-service";
import { buildOverlay, type Overlay, type OverlayInput } from "./watchlist-overlay";

const MAX_TICKERS = 10;

export async function getWatchlistOverlay(): Promise<{ tickers: string[]; overlay: Overlay }> {
  const rows = await getWatchlist();
  const tickers = rows.slice(0, MAX_TICKERS).map((r) => r.ticker);
  if (tickers.length === 0) return { tickers: [], overlay: { dates: [], series: [] } };

  const datas = await Promise.all(
    tickers.map((t) =>
      getTickerData(t, "1y")
        .then((d): OverlayInput => ({ ticker: d.ticker, bars: d.bars }))
        .catch(() => null),
    ),
  );
  const items = datas.filter((d): d is OverlayInput => d !== null);
  return { tickers, overlay: buildOverlay(items) };
}
