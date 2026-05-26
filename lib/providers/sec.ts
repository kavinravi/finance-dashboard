import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { RawCompanyFacts } from "@/lib/types";

const TICKERS_URL = "https://www.sec.gov/files/company_tickers.json";
const FACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts";
// Not hard-gated — SEC has no daily cap (only a ~10 req/s guideline that caching respects).
// The value only feeds the provider_state health row.
export const SEC_DAILY_LIMIT = 100000;

type TickerMap = Record<string, { cik_str: number; ticker: string; title: string }>;

export function findCik(map: TickerMap, ticker: string): string | null {
  const upper = ticker.toUpperCase();
  for (const k in map) {
    const entry = map[k];
    if (entry?.ticker?.toUpperCase() === upper) return String(entry.cik_str).padStart(10, "0");
  }
  return null;
}

export const sec = {
  name: "sec",
  async resolveCik(ticker: string): Promise<string | null> {
    if (!env.SEC_USER_AGENT) return null;
    try {
      const res = await fetch(TICKERS_URL, { headers: { "User-Agent": env.SEC_USER_AGENT } });
      if (!res.ok) throw new Error(`SEC tickers ${res.status}`);
      const cik = findCik((await res.json()) as TickerMap, ticker);
      await recordSuccess("sec", SEC_DAILY_LIMIT);
      return cik;
    } catch (e) {
      await recordError("sec", SEC_DAILY_LIMIT, String(e));
      return null;
    }
  },
  async fetchCompanyFacts(cik: string): Promise<RawCompanyFacts | null> {
    if (!env.SEC_USER_AGENT) return null;
    const url = `${FACTS_BASE}/CIK${cik}.json`;
    try {
      const res = await fetch(url, { headers: { "User-Agent": env.SEC_USER_AGENT } });
      if (!res.ok) throw new Error(`SEC facts ${res.status}`);
      const data = (await res.json()) as RawCompanyFacts;
      await recordSuccess("sec", SEC_DAILY_LIMIT);
      return data;
    } catch (e) {
      await recordError("sec", SEC_DAILY_LIMIT, String(e));
      return null;
    }
  },
};
