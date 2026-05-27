"use client";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { ToneMeter } from "@/components/tone-meter";
import { NEWS_WINDOWS, DEFAULT_WINDOW } from "@/lib/news/windows";
import type { MemoResult, CitedArticle } from "@/lib/services/memo-service";

type Development = { claim: string; why_it_matters: string; source_article_ids: string[]; confidence: string };

function Chips({ ids, cited }: { ids: string[]; cited: CitedArticle[] }) {
  return (
    <>
      {ids.map((id) => {
        const idx = cited.findIndex((c) => c.id === id);
        const a = cited[idx];
        if (!a) return null;
        return (
          <a key={id} href={a.url} target="_blank" rel="noopener noreferrer"
            title={a.title}
            className="ml-1 rounded bg-neutral-800 px-1 text-xs text-sky-300 hover:bg-neutral-700">
            {idx + 1}
          </a>
        );
      })}
    </>
  );
}

function Group({ title, items, cited, tone }: { title: string; items: Development[]; cited: CitedArticle[]; tone: "bullish" | "bearish" | "neutral" }) {
  if (items.length === 0) return null;
  const titleColor = tone === "bullish" ? "text-emerald-400" : tone === "bearish" ? "text-red-400" : "text-neutral-500";
  return (
    <div className="mt-4">
      <h3 className={`text-xs font-medium uppercase ${titleColor}`}>{title}</h3>
      <ul className="mt-1 space-y-2">
        {items.map((d, i) => (
          <li key={i} className="text-sm">
            <span className="text-neutral-100">{d.claim}</span>
            <span className="text-neutral-500"> — {d.why_it_matters}</span>
            <span className="ml-1 text-xs text-neutral-600">({d.confidence})</span>
            <Chips ids={d.source_article_ids} cited={cited} />
          </li>
        ))}
      </ul>
    </div>
  );
}

export function MemoCard({ symbol }: { symbol: string }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<MemoResult | null>(null);
  const [days, setDays] = useState<number>(DEFAULT_WINDOW);

  const load = useCallback(async (force: boolean) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/memo/${symbol}?days=${days}${force ? "&force=1" : ""}`);
      setData(await res.json());
    } catch {
      setData({ status: "error", memo: null, citedArticles: [] });
    } finally {
      setLoading(false);
    }
  }, [symbol, days]);

  useEffect(() => { void load(false); }, [load]);

  const windowToggle = (
    <div className="mb-2 flex items-center gap-1 text-xs">
      <span className="mr-1 text-neutral-500">News window:</span>
      {NEWS_WINDOWS.map((w) => (
        <button key={w} onClick={() => setDays(w)}
          className={`rounded px-2 py-0.5 ${days === w ? "bg-neutral-200 text-neutral-900" : "bg-neutral-900 text-neutral-300 ring-1 ring-neutral-800 hover:bg-neutral-800"}`}>
          {w}d
        </button>
      ))}
    </div>
  );

  let body: ReactNode;
  if (loading) {
    body = <p className="text-sm text-neutral-500">Generating today&apos;s memo…</p>;
  } else if (!data || data.status === "error") {
    body = (
      <div className="text-sm text-neutral-500">
        Couldn&apos;t generate the memo. <button onClick={() => load(true)} className="underline">Try again</button>
      </div>
    );
  } else if (data.status === "no_news") {
    body = <p className="text-sm text-neutral-500">No recent news in the last {days} day{days === 1 ? "" : "s"}.</p>;
  } else if (data.status === "unavailable") {
    body = <p className="text-sm text-neutral-500">Memo unavailable — Gemini key missing or daily limit reached.</p>;
  } else if (!data.memo) {
    body = (
      <div className="text-sm text-neutral-500">
        Couldn&apos;t render the memo. <button onClick={() => load(true)} className="underline">Try again</button>
      </div>
    );
  } else {
    const m = data.memo;
    body = (
      <div className="rounded-lg ring-1 ring-neutral-800 p-4">
        <p className="text-base text-neutral-100">{m.one_sentence_takeaway}</p>
        <ToneMeter label={m.overall_news_tone.label} score={m.overall_news_tone.score} />
        <Group title="Bullish" items={m.bullish_developments} cited={data.citedArticles} tone="bullish" />
        <Group title="Bearish" items={m.bearish_developments} cited={data.citedArticles} tone="bearish" />
        <Group title="Neutral / operational" items={m.neutral_or_operational_updates} cited={data.citedArticles} tone="neutral" />
        {m.watch_items.length > 0 && (
          <div className="mt-4">
            <h3 className="text-xs font-medium uppercase text-neutral-500">Watch items</h3>
            <ul className="mt-1 list-disc pl-5 text-sm text-neutral-300">{m.watch_items.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </div>
        )}
        {m.caveats.length > 0 && (
          <div className="mt-4">
            <h3 className="text-xs font-medium uppercase text-neutral-500">Caveats</h3>
            <ul className="mt-1 list-disc pl-5 text-sm text-neutral-400">{m.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
          </div>
        )}
        <div className="mt-4 flex items-center justify-between border-t border-neutral-800 pt-2 text-xs text-neutral-600">
          <span>
            Generated by {m.model} from {m.basedOnArticleCount} sources over the last {days}d · every claim links to its source · News Tone reflects coverage tone, not a forecast · not investment advice.
          </span>
          <button onClick={() => load(true)} className="ml-3 shrink-0 underline">Regenerate</button>
        </div>
      </div>
    );
  }

  return (
    <div>
      {windowToggle}
      {body}
    </div>
  );
}
