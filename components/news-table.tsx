import type { ArticleRow } from "@/lib/db/articles";

function relTime(d: Date): string {
  const mins = Math.round((Date.now() - d.getTime()) / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

const sourceName = (s: string) => (s === "finnhub" ? "Finnhub" : "Yahoo");

export function NewsTable({ articles }: { articles: ArticleRow[] }) {
  if (articles.length === 0) {
    return <p className="text-sm text-neutral-500">No recent news.</p>;
  }
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="text-left text-xs uppercase text-neutral-500">
          <th className="py-1 pr-3 font-medium">Time</th>
          <th className="py-1 pr-3 font-medium">Source</th>
          <th className="py-1 font-medium">Headline</th>
        </tr>
      </thead>
      <tbody>
        {articles.map((a) => (
          <tr key={a.id} className="border-t border-neutral-800 align-top">
            <td className="whitespace-nowrap py-2 pr-3 text-neutral-400" title={a.publishedAt.toISOString()}>
              {relTime(a.publishedAt)}
            </td>
            <td className="whitespace-nowrap py-2 pr-3 text-neutral-400">{sourceName(a.source)}</td>
            <td className="py-2">
              <a href={a.url} target="_blank" rel="noopener noreferrer" className="text-neutral-100 hover:underline">
                {a.title}
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
