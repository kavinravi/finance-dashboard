import { getTickerData } from "@/lib/services/price-service";
import { getNews } from "@/lib/services/news-service";
import { MemoCard } from "@/components/memo-card";
import { NewsTable } from "@/components/news-table";

export const dynamic = "force-dynamic";

export default async function TickerNewsPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const ticker = symbol.toUpperCase();
  await getTickerData(ticker, "1y"); // ensure the company row exists for deep links
  const news = await getNews(ticker);

  return (
    <div className="mt-6">
      <section>
        <h2 className="text-sm font-medium text-neutral-400">Daily memo</h2>
        <p className="mb-2 text-xs text-neutral-600">Research assistant, not investment advice.</p>
        <MemoCard symbol={ticker} />
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Recent news</h2>
        <div className="mt-2"><NewsTable articles={news.articles} /></div>
      </section>
    </div>
  );
}
