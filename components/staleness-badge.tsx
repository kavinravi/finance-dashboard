export function StalenessBadge({ stale, lastBarDate, source }: { stale: boolean; lastBarDate: string | null; source: string }) {
  return (
    <span className={`rounded px-2 py-0.5 text-xs ${stale ? "bg-amber-900 text-amber-200" : "bg-neutral-800 text-neutral-400"}`}>
      {lastBarDate ? `data as of ${lastBarDate}` : "no data"} · {source}{stale ? " · stale" : ""}
    </span>
  );
}
