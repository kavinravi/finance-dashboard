import Link from "next/link";
import { TickerTabs } from "@/components/ticker-tabs";

export default async function TickerLayout({
  children, params,
}: { children: React.ReactNode; params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const ticker = symbol.toUpperCase();
  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href="/" className="text-sm text-neutral-500">← Search</Link>
      <h1 className="mt-4 font-mono text-3xl font-semibold">{ticker}</h1>
      <TickerTabs symbol={ticker} />
      {children}
    </main>
  );
}
