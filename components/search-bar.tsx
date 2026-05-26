"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import type { SearchResult } from "@/lib/types";

export function SearchBar() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      setResults(data.results ?? []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-xl">
      <form onSubmit={run} className="flex gap-2">
        <input
          value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="Search ticker or company (e.g. NVDA, NVIDIA)"
          className="flex-1 rounded bg-neutral-900 px-3 py-2 outline-none ring-1 ring-neutral-800 focus:ring-neutral-600"
        />
        <button className="rounded bg-neutral-200 px-4 py-2 font-medium text-neutral-900" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>
      {results.length > 0 && (
        <ul className="mt-2 divide-y divide-neutral-800 rounded bg-neutral-900 ring-1 ring-neutral-800">
          {results.slice(0, 8).map((r) => (
            <li key={`${r.symbol}-${r.source}`}>
              <button
                onClick={() => router.push(`/ticker/${r.symbol}`)}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-neutral-800"
              >
                <span><span className="font-mono font-semibold">{r.symbol}</span> · {r.name}</span>
                <span className="text-xs text-neutral-500">{r.exchange ?? ""}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
