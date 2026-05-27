"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

type Profile = { id: string; name: string };
const TILE_COLORS = ["#2563eb", "#dc2626", "#d4a017", "#0f766e", "#7c3aed", "#db2777"];

export function ProfilePicker({ profiles, next }: { profiles: Profile[]; next: string }) {
  const router = useRouter();
  const [managing, setManaging] = useState(false);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function select(id: string) {
    setBusy(true);
    try {
      const res = await fetch("/api/profile/select", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, next }),
      });
      if (!res.ok) { setError("Couldn't switch profile. Try again."); setBusy(false); return; }
      const data = await res.json().catch(() => ({ next }));
      window.location.href = data.next ?? next; // full reload so server data re-scopes to the chosen profile
    } catch {
      setError("Couldn't switch profile. Try again.");
      setBusy(false);
    }
  }

  async function addProfile(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/profiles", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: trimmed }),
      });
      const data = await res.json().catch(() => null);
      if (data?.ok) { await select(data.profile.id); return; } // select() navigates away on success
      setError(data?.error === "duplicate" ? `"${trimmed}" already exists.` : "Couldn't create profile.");
      setBusy(false); // leave the form open so the user can fix the name
    } catch {
      setError("Couldn't create profile.");
      setBusy(false);
    }
  }

  async function remove(id: string, label: string) {
    if (!confirm(`Delete profile "${label}" and its watchlist?`)) return; // destructive: cascades the watchlist
    setBusy(true);
    try {
      await fetch(`/api/profiles/${id}`, { method: "DELETE" });
      router.refresh(); // re-render the server list
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="flex flex-wrap items-start justify-center gap-6">
        {profiles.map((p, i) => (
          <div key={p.id} className="relative flex w-28 flex-col items-center gap-2">
            <button
              type="button"
              onClick={() => select(p.id)}
              disabled={busy || managing}
              className="h-28 w-28 rounded-md ring-2 ring-transparent transition hover:ring-white disabled:opacity-60"
              style={{ background: TILE_COLORS[i % TILE_COLORS.length] }}
              aria-label={p.name}
            />
            {managing && (
              <button
                type="button"
                aria-label={`Delete ${p.name}`}
                onClick={() => remove(p.id, p.name)}
                disabled={busy}
                className="absolute right-1 top-1 rounded bg-black/60 px-1 text-xs text-white hover:bg-black disabled:opacity-60"
              >×</button>
            )}
            <span className="text-sm text-neutral-300">{p.name}</span>
          </div>
        ))}

        <div className="flex w-28 flex-col items-center gap-2">
          {adding ? (
            <form onSubmit={addProfile} className="flex h-28 w-28 flex-col items-center justify-center gap-1 rounded-md ring-1 ring-neutral-700">
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} maxLength={40} placeholder="Name"
                className="w-24 rounded bg-neutral-900 px-1 py-0.5 text-center text-sm text-neutral-100 ring-1 ring-neutral-700" />
              <button type="submit" disabled={busy} className="text-xs text-neutral-300 hover:text-white disabled:opacity-60">Add</button>
            </form>
          ) : (
            <button type="button" onClick={() => { setAdding(true); setError(""); }} disabled={busy}
              className="flex h-28 w-28 items-center justify-center rounded-md text-4xl text-neutral-500 ring-1 ring-neutral-700 hover:text-white hover:ring-white disabled:opacity-60">+</button>
          )}
          <span className="text-sm text-neutral-500">Add Profile</span>
        </div>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {profiles.length > 0 && (
        <button type="button" onClick={() => setManaging((m) => !m)}
          className="rounded border border-neutral-600 px-4 py-2 text-sm uppercase tracking-wider text-neutral-300 hover:border-white hover:text-white">
          {managing ? "Done" : "Manage Profiles"}
        </button>
      )}
    </div>
  );
}
