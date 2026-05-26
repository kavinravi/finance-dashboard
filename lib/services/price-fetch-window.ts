import type { PriceBar } from "@/lib/types";

export const HISTORY_FLOOR = "1970-01-01";
const DEEP_YEARS = 3;
const REVISION_BUFFER_DAYS = 5;

// Decides the `from` date for a refetch.
// - Cold cache, or a cache that doesn't yet reach back DEEP_YEARS → fetch full history (so "ALL" really means all,
//   and pre-existing shallow (e.g. 2-year) caches backfill on their next refresh).
// - Once we hold bars older than DEEP_YEARS → refetch incrementally from a few days before the last bar.
export function fetchFromDate(bars: PriceBar[], today: string): string {
  const deepCutoff = new Date(`${today}T00:00:00Z`);
  deepCutoff.setUTCFullYear(deepCutoff.getUTCFullYear() - DEEP_YEARS);
  const deepCutoffIso = deepCutoff.toISOString().slice(0, 10);

  if (bars.length === 0 || bars[0].date > deepCutoffIso) return HISTORY_FLOOR;

  const last = new Date(`${bars[bars.length - 1].date}T00:00:00Z`);
  last.setUTCDate(last.getUTCDate() - REVISION_BUFFER_DAYS);
  return last.toISOString().slice(0, 10);
}
