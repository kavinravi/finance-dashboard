import Link from "next/link";
import { listRecentSearches } from "@/lib/db/recent-searches";

export async function RecentSearches() {
  const rows = await listRecentSearches(8);
  const resolved = rows.filter((r) => r.resolvedTicker);
  if (resolved.length === 0) return null;
  return (
    <div className="mt-8 w-full max-w-xl">
      <h2 className="mb-2 text-xs uppercase tracking-wide text-neutral-500">Recent</h2>
      <div className="flex flex-wrap gap-2">
        {resolved.map((r) => (
          <Link key={r.id} href={`/ticker/${r.resolvedTicker}`}
            className="rounded bg-neutral-900 px-3 py-1 font-mono text-sm ring-1 ring-neutral-800 hover:bg-neutral-800">
            {r.resolvedTicker}
          </Link>
        ))}
      </div>
    </div>
  );
}
