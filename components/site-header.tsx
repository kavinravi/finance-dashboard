"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { SearchBar } from "./search-bar";
import { ProfileSwitcher } from "./profile-switcher";

export function SiteHeader() {
  const pathname = usePathname();
  if (pathname === "/login" || pathname === "/select-profile") return null;
  const showSearch = pathname !== "/";

  async function logout() {
    await fetch("/api/logout", { method: "POST" });
    window.location.href = "/login";
  }

  return (
    <header className="flex items-center gap-4 border-b border-neutral-800 px-4 py-2 text-sm">
      {showSearch ? <div className="flex-1"><SearchBar /></div> : <div className="flex-1" />}
      <nav className="flex items-center gap-4">
        <Link href="/" className="text-neutral-300 hover:text-white">Home</Link>
        <Link href="/watchlist" className="text-neutral-300 hover:text-white">Watchlist</Link>
        <Link href="/health" className="text-neutral-300 hover:text-white">Health</Link>
        <ProfileSwitcher />
        <button onClick={logout} className="text-neutral-300 hover:text-white">Log out</button>
      </nav>
    </header>
  );
}
