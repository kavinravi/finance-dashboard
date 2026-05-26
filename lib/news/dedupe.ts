import { createHash } from "node:crypto";
import type { NewsArticle } from "@/lib/types";

const TRACKING = new Set(["fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src"]);

export function canonicalizeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    u.hash = "";
    u.hostname = u.hostname.toLowerCase().replace(/^www\./, "");
    for (const k of [...u.searchParams.keys()]) {
      if (/^utm_/i.test(k) || TRACKING.has(k.toLowerCase())) u.searchParams.delete(k);
    }
    u.pathname = u.pathname.replace(/\/+$/, "");
    return u.toString();
  } catch {
    return raw.trim();
  }
}

export function urlHash(raw: string): string {
  return createHash("sha256").update(canonicalizeUrl(raw)).digest("hex");
}

export function normalizeTitle(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
}

export function dedupeArticles(articles: NewsArticle[]): NewsArticle[] {
  const seenHash = new Set<string>();
  const seenTitle = new Set<string>();
  const out: NewsArticle[] = [];
  const sorted = [...articles].sort((a, b) => b.publishedAt.getTime() - a.publishedAt.getTime());
  for (const a of sorted) {
    const h = urlHash(a.url);
    const t = normalizeTitle(a.title);
    if (seenHash.has(h) || (t.length > 0 && seenTitle.has(t))) continue;
    seenHash.add(h);
    if (t.length > 0) seenTitle.add(t);
    out.push(a);
  }
  return out;
}
