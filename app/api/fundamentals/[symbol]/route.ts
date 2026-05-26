import { type NextRequest } from "next/server";
import { z } from "zod";
import { getFundamentals } from "@/lib/services/fundamentals-service";

export const dynamic = "force-dynamic";
export const maxDuration = 30; // SEC companyfacts fetch can be a few MB

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);

export async function GET(_request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  try {
    return Response.json(await getFundamentals(t.data.toUpperCase()));
  } catch (e) {
    // Surface as a card-renderable error rather than an HTTP failure.
    return Response.json({ status: "error", view: null, asOf: null, source: null, detail: String(e) });
  }
}
