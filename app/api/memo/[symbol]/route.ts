import { type NextRequest } from "next/server";
import { z } from "zod";
import { getMemo } from "@/lib/services/memo-service";
import { parseWindow } from "@/lib/news/windows";

export const dynamic = "force-dynamic";
export const maxDuration = 60; // news fetch + Gemini can be slow

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);

export async function GET(request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  const force = request.nextUrl.searchParams.get("force") === "1";
  const lookbackDays = parseWindow(request.nextUrl.searchParams.get("days"));
  try {
    return Response.json(await getMemo(t.data.toUpperCase(), { force, lookbackDays }));
  } catch (e) {
    // Surface as a card-renderable error state rather than an HTTP failure.
    return Response.json({ status: "error", memo: null, citedArticles: [], detail: String(e) });
  }
}
