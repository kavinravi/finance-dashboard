"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";

type Profile = { id: string; name: string };

export function ProfileSwitcher() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/profiles")
      .then((r) => r.json())
      .then((d) => { setProfiles(d.profiles ?? []); setActiveId(d.activeId ?? null); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  async function switchTo(id: string) {
    await fetch("/api/profile/select", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ id }),
    });
    window.location.reload(); // re-scope server-rendered data to the new profile
  }

  const active = profiles.find((p) => p.id === activeId);

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen((o) => !o)} className="rounded px-2 py-1 text-neutral-300 ring-1 ring-neutral-800 hover:text-white">
        {active ? active.name : "Profile"} ▾
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-1 w-44 rounded bg-neutral-900 py-1 text-sm shadow-lg ring-1 ring-neutral-800">
          {profiles.map((p) => (
            <button key={p.id} onClick={() => switchTo(p.id)}
              className={`block w-full px-3 py-1 text-left hover:bg-neutral-800 ${p.id === activeId ? "text-white" : "text-neutral-300"}`}>
              {p.name}
            </button>
          ))}
          <Link href="/select-profile" className="mt-1 block border-t border-neutral-800 px-3 py-1 text-neutral-400 hover:bg-neutral-800 hover:text-white">
            Manage profiles
          </Link>
        </div>
      )}
    </div>
  );
}
