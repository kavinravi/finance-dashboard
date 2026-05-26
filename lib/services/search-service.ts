import { fmp } from "@/lib/providers/fmp";
import { yahoo } from "@/lib/providers/yahoo";
import { logSearch } from "@/lib/db/recent-searches";
import type { SearchResult } from "@/lib/types";

export async function resolveQuery(query: string): Promise<SearchResult[]> {
  const q = query.trim();
  if (!q) return [];

  let results: SearchResult[] = [];
  try {
    results = await fmp.search(q);
  } catch {
    try {
      results = await yahoo.search(q);
    } catch {
      results = [];
    }
  }
  await logSearch(q, results[0]?.symbol ?? null);
  return results;
}
