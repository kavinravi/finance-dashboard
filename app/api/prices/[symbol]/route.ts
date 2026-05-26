import { type NextRequest } from "next/server";
import { z } from "zod";
import { getTickerData } from "@/lib/services/price-service";

const Ticker = z.string().regex(/^[A-Za-z.\-]{1,10}$/);
const Range = z.enum(["1m", "3m", "6m", "ytd", "1y"]).default("1y");

export async function GET(request: NextRequest, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const t = Ticker.safeParse(symbol);
  if (!t.success) return Response.json({ error: "Invalid ticker" }, { status: 400 });
  const range = Range.parse(request.nextUrl.searchParams.get("range") ?? undefined);
  try {
    return Response.json(await getTickerData(t.data.toUpperCase(), range));
  } catch (e) {
    return Response.json({ error: "Price lookup failed", detail: String(e) }, { status: 502 });
  }
}
