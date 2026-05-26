"use client";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export function WatchlistManager({ initialTickers }: { initialTickers: string[] }) {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    const ticker = input.trim().toUpperCase();
    if (!ticker) return;
    setBusy(true);
    try {
      await fetch("/api/watchlist", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ticker }),
      });
      setInput("");
      router.refresh();
    } finally {
      setBusy(false);
    }
  }

  async function remove(ticker: string) {
    setBusy(true);
    try {
      await fetch("/api/watchlist", {
        method: "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ ticker }),
      });
      router.refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <form onSubmit={add} className="flex items-center gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Add ticker (e.g. MSFT)"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button disabled={busy} className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900 disabled:opacity-50">Add</button>
      </form>
      <div className="flex flex-wrap gap-2">
        {initialTickers.map((t) => (
          <span key={t} className="flex items-center gap-1 rounded bg-neutral-800 px-2 py-1 text-sm">
            <Link href={`/ticker/${t}`} className="font-mono hover:underline">{t}</Link>
            <button onClick={() => remove(t)} disabled={busy} aria-label={`Remove ${t}`} className="text-neutral-500 hover:text-red-400">×</button>
          </span>
        ))}
      </div>
    </div>
  );
}
