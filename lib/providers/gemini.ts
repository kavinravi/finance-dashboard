import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";

export const developmentSchema = z.object({
  claim: z.string(),
  why_it_matters: z.string(),
  source_article_ids: z.array(z.string()),
  confidence: z.enum(["low", "medium", "high"]),
});

export const memoOutputSchema = z.object({
  ticker: z.string(),
  date: z.string(),
  one_sentence_takeaway: z.string(),
  bullish_developments: z.array(developmentSchema),
  bearish_developments: z.array(developmentSchema),
  neutral_or_operational_updates: z.array(developmentSchema),
  watch_items: z.array(z.string()),
  caveats: z.array(z.string()),
  overall_news_tone: z.object({
    label: z.enum(["bearish", "somewhat_bearish", "neutral", "somewhat_bullish", "bullish"]),
    score: z.number().int().min(0).max(100),
    rationale: z.string(),
  }),
});
export type MemoOutput = z.infer<typeof memoOutputSchema>;

export type MemoInputArticle = {
  id: string; source: string; publishedAt: string; headline: string; summary: string | null; related: string | null;
};
export type MemoInput = {
  ticker: string; companyName: string; date: string;
  priceContext: {
    latestClose: number | null; currency: string | null;
    returns: { d1: number | null; d5: number | null; m1: number | null; y1: number | null };
  };
  articles: MemoInputArticle[];
};

const SHAPE_EXAMPLE = {
  ticker: "TICK", date: "YYYY-MM-DD", one_sentence_takeaway: "string",
  bullish_developments: [{ claim: "string", why_it_matters: "string", source_article_ids: ["a1"], confidence: "low|medium|high" }],
  bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: ["string"], caveats: ["string"],
  overall_news_tone: { label: "bearish|somewhat_bearish|neutral|somewhat_bullish|bullish", score: 0, rationale: "string" },
};

function pct(v: number | null): string {
  return v === null || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(2)}%`;
}

export function buildPrompt(input: MemoInput): string {
  const pc = input.priceContext;
  return [
    `You are a financial research assistant. Produce a NEWS TONE memo for ${input.ticker} (${input.companyName}) dated ${input.date}.`,
    `You are NOT an advisor. Never output buy, sell, hold, or price targets.`,
    ``,
    `Rules:`,
    `- Use ONLY the articles listed below. Do not invent facts or imply access to full article bodies.`,
    `- Every development MUST cite at least one article id (e.g. "a1") from the provided set in source_article_ids.`,
    `- Separate confirmed company events from analyst speculation.`,
    `- If evidence is thin, duplicated, or stale, say so in caveats, lower confidence, and return mostly-empty arrays.`,
    `- overall_news_tone reflects the tone of COVERAGE, not a stock forecast; rationale must reference the actual articles.`,
    ``,
    `Price context (factual; do NOT speculate on causation): latest close ${pc.latestClose ?? "n/a"} ${pc.currency ?? ""}; returns 1D ${pct(pc.returns.d1)}, 5D ${pct(pc.returns.d5)}, 1M ${pct(pc.returns.m1)}, 1Y ${pct(pc.returns.y1)}.`,
    ``,
    `Articles:`,
    ...input.articles.map((a) => `[${a.id}] (${a.source}, ${a.publishedAt}) ${a.headline}${a.summary ? ` — ${a.summary}` : ""}`),
    ``,
    `Respond with ONLY a JSON object (no markdown fences) matching this shape exactly:`,
    JSON.stringify(SHAPE_EXAMPLE, null, 2),
  ].join("\n");
}

export async function generateMemo(input: MemoInput): Promise<MemoOutput> {
  const apiKey = env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("GEMINI_API_KEY is not set");
  const ai = new GoogleGenAI({ apiKey });
  let lastErr = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await ai.models.generateContent({
        model: env.GEMINI_MODEL,
        contents: buildPrompt(input),
        config: { responseMimeType: "application/json", temperature: 0.2 },
      });
      const parsed = memoOutputSchema.safeParse(JSON.parse(res.text ?? ""));
      if (parsed.success) {
        await recordSuccess("gemini", env.GEMINI_DAILY_LIMIT);
        return parsed.data;
      }
      lastErr = "schema validation failed: " + parsed.error.issues.map((i) => i.path.join(".") + " " + i.message).join("; ");
    } catch (e) {
      lastErr = String(e);
    }
  }
  await recordError("gemini", env.GEMINI_DAILY_LIMIT, lastErr);
  throw new Error(`Gemini memo generation failed: ${lastErr}`);
}
