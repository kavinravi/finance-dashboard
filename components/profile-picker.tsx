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

  async function select(id: string) {
    setBusy(true);
    const res = await fetch("/api/profile/select", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, next }),
    });
    const data = await res.json().catch(() => ({ next }));
    window.location.href = data.next ?? next; // full reload so server data re-scopes to the chosen profile
  }

  async function addProfile(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    const res = await fetch("/api/profiles", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: trimmed }),
    });
    const data = await res.json().catch(() => null);
    if (data?.ok) { await select(data.profile.id); return; }
    setBusy(false);
    setAdding(false);
    setName("");
  }

  async function remove(id: string, label: string) {
    if (!confirm(`Delete profile "${label}" and its watchlist?`)) return; // destructive: cascades the watchlist
    setBusy(true);
    await fetch(`/api/profiles/${id}`, { method: "DELETE" });
    setBusy(false);
    router.refresh(); // re-render the server list
  }

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="flex flex-wrap items-start justify-center gap-6">
        {profiles.map((p, i) => (
          <div key={p.id} className="flex w-28 flex-col items-center gap-2">
            <button
              onClick={() => { if (!managing) select(p.id); }}
              disabled={busy}
              className="relative h-28 w-28 rounded-md ring-2 ring-transparent transition hover:ring-white disabled:opacity-50"
              style={{ background: TILE_COLORS[i % TILE_COLORS.length] }}
              aria-label={p.name}
            >
              {managing && (
                <span
                  role="button"
                  aria-label={`Delete ${p.name}`}
                  onClick={(e) => { e.stopPropagation(); remove(p.id, p.name); }}
                  className="absolute right-1 top-1 rounded bg-black/60 px-1 text-xs text-white hover:bg-black"
                >×</span>
              )}
            </button>
            <span className="text-sm text-neutral-300">{p.name}</span>
          </div>
        ))}

        <div className="flex w-28 flex-col items-center gap-2">
          {adding ? (
            <form onSubmit={addProfile} className="flex h-28 w-28 flex-col items-center justify-center gap-1 rounded-md ring-1 ring-neutral-700">
              <input autoFocus value={name} onChange={(e) => setName(e.target.value)} maxLength={40} placeholder="Name"
                className="w-24 rounded bg-neutral-900 px-1 py-0.5 text-center text-sm text-neutral-100 ring-1 ring-neutral-700" />
              <button disabled={busy} className="text-xs text-neutral-300 hover:text-white">Add</button>
            </form>
          ) : (
            <button onClick={() => setAdding(true)} disabled={busy}
              className="flex h-28 w-28 items-center justify-center rounded-md text-4xl text-neutral-500 ring-1 ring-neutral-700 hover:text-white hover:ring-white">+</button>
          )}
          <span className="text-sm text-neutral-500">Add Profile</span>
        </div>
      </div>

      {profiles.length > 0 && (
        <button onClick={() => setManaging((m) => !m)}
          className="rounded border border-neutral-600 px-4 py-2 text-sm uppercase tracking-wider text-neutral-300 hover:border-white hover:text-white">
          {managing ? "Done" : "Manage Profiles"}
        </button>
      )}
    </div>
  );
}
