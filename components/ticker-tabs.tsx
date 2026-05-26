"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function TickerTabs({ symbol }: { symbol: string }) {
  const pathname = usePathname();
  const onNews = pathname.endsWith("/news");
  const base = `/ticker/${symbol}`;
  const cls = (active: boolean) =>
    `pb-2 text-sm ${active ? "border-b-2 border-neutral-200 text-neutral-100" : "text-neutral-500 hover:text-neutral-300"}`;
  return (
    <nav className="mt-4 flex gap-6 border-b border-neutral-800">
      <Link href={base} className={cls(!onNews)}>Charts &amp; Fundamentals</Link>
      <Link href={`${base}/news`} className={cls(onNews)}>News &amp; Memo</Link>
      <Link href="/watchlist" className={cls(false)}>Watchlist</Link>
    </nav>
  );
}
