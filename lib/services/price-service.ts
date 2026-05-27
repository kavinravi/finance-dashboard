import { env } from "@/lib/env";
import { fmp } from "@/lib/providers/fmp";
import { yahoo } from "@/lib/providers/yahoo";
import { getCompanyByTicker, upsertCompany } from "@/lib/db/companies";
import { getBars, upsertBars } from "@/lib/db/price-bars";
import { canCall, recordSuccess, recordError } from "@/lib/db/provider-state";
import { computeReturns, sma, rsi, macd, rollingVolatility } from "@/lib/indicators";
import type { PriceBar, PeriodReturns } from "@/lib/types";
import { fetchFromDate } from "@/lib/services/price-fetch-window";

export type Range = "1m" | "3m" | "6m" | "ytd" | "1y";

export type TickerData = {
  ticker: string;
  bars: PriceBar[];
  returns: PeriodReturns;
  indicators: {
    ma10: (number | null)[]; ma20: (number | null)[]; ma50: (number | null)[]; ma200: (number | null)[];
    rsi14: (number | null)[];
    macdLine: number[]; macdSignal: number[]; macdHistogram: number[];
    volatility5d: (number | null)[];
  };
  source: "cache" | "fmp" | "yahoo";
  lastBarDate: string | null;
  stale: boolean;
};

function lastTradingDayIso(): string {
  const d = new Date();
  const day = d.getUTCDay();
  if (day === 0) d.setUTCDate(d.getUTCDate() - 2);
  else if (day === 6) d.setUTCDate(d.getUTCDate() - 1);
  return d.toISOString().slice(0, 10);
}

function isFresh(bars: PriceBar[]): boolean {
  if (bars.length === 0) return false;
  return bars[bars.length - 1].date >= lastTradingDayIso();
}

async function ensureCompany(ticker: string) {
  const existing = await getCompanyByTicker(ticker);
  if (existing) return existing;
  const profile = await fmp.profile(ticker).catch(() => null);
  return upsertCompany(
    profile ?? { ticker, name: ticker, assetType: "stock", exchange: null, sector: null, industry: null, currency: null },
  );
}

// History is fetched max-available (see fetchFromDate). `range` is applied client-side (lib/charts/range.ts),
// so it stays accepted-but-unused here to keep callers (e.g. comparison-service) unchanged.
export async function getTickerData(ticker: string, _range: Range = "1y"): Promise<TickerData> {
  const company = await ensureCompany(ticker);
  let bars = await getBars(company.id);
  let source: TickerData["source"] = "cache";

  if (!isFresh(bars)) {
    const toIso = new Date().toISOString().slice(0, 10);
    const fromIso = fetchFromDate(bars, toIso);

    let fetched: PriceBar[] | null = null;
    if (await canCall("fmp", env.FMP_DAILY_LIMIT)) {
      try {
        fetched = await fmp.dailyPrices(company.ticker, fromIso, toIso);
      } catch (e) {
        await recordError("fmp", env.FMP_DAILY_LIMIT, String(e));
        fetched = null;
      }
      // Record success + persist outside the fetch try so a DB hiccup here
      // can't masquerade as an FMP failure and discard good data.
      if (fetched && fetched.length) {
        await recordSuccess("fmp", env.FMP_DAILY_LIMIT);
        await upsertBars(company.id, fetched, "fmp");
        source = "fmp";
      }
    }
    if (fetched === null || fetched.length === 0) {
      try {
        const y = await yahoo.dailyPrices(company.ticker, fromIso, toIso);
        if (y.length) { await upsertBars(company.id, y, "yahoo"); source = "yahoo"; }
      } catch { /* degrade to cache */ }
    }
    bars = await getBars(company.id);
  }

  const closes = bars.map((b) => b.close);
  const lastBarDate = bars.at(-1)?.date ?? null;
  const m = macd(closes);
  return {
    ticker: company.ticker,
    bars,
    returns: computeReturns(bars),
    indicators: {
      ma10: sma(closes, 10), ma20: sma(closes, 20), ma50: sma(closes, 50), ma200: sma(closes, 200),
      rsi14: rsi(closes, 14),
      macdLine: m.macdLine, macdSignal: m.signalLine, macdHistogram: m.histogram,
      volatility5d: rollingVolatility(closes, 5),
    },
    source,
    lastBarDate,
    stale: lastBarDate !== null && lastBarDate < lastTradingDayIso(),
  };
}
