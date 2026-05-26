"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function AppNav() {
  const pathname = usePathname();
  if (pathname === "/login") return null;

  async function logout() {
    await fetch("/api/logout", { method: "POST" });
    window.location.href = "/login";
  }

  return (
    <nav className="flex items-center justify-end gap-4 border-b border-neutral-800 px-4 py-2 text-sm">
      <Link href="/" className="text-neutral-300 hover:text-white">Home</Link>
      <Link href="/watchlist" className="text-neutral-300 hover:text-white">Watchlist</Link>
      <Link href="/health" className="text-neutral-300 hover:text-white">Health</Link>
      <button onClick={logout} className="text-neutral-300 hover:text-white">Log out</button>
    </nav>
  );
}
