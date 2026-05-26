"use client";
import { useState } from "react";

export function PruneButton() {
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onClick() {
    setLoading(true);
    setMsg(null);
    const res = await fetch("/api/admin/prune", { method: "POST" });
    setLoading(false);
    if (res.ok) {
      const d = await res.json();
      setMsg(`Removed ${d.articles} articles, ${d.fundamentals} snapshots.`);
    } else {
      setMsg("Prune failed.");
    }
  }

  return (
    <div className="space-y-2">
      <button
        onClick={onClick}
        disabled={loading}
        className="rounded bg-neutral-800 px-3 py-1.5 text-sm hover:bg-neutral-700 disabled:opacity-50"
      >
        {loading ? "Pruning…" : "Prune now"}
      </button>
      {msg && <p className="text-sm text-neutral-400">{msg}</p>}
    </div>
  );
}
