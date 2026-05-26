import { NextResponse } from "next/server";
import { z } from "zod";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const schema = z.object({ ticker: z.string().regex(/^[A-Za-z.\-]{1,10}$/) });
const list = async () => ({ tickers: (await getWatchlist()).map((w) => w.ticker) });

export async function GET() {
  return NextResponse.json(await list());
}

export async function POST(req: Request) {
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await addToWatchlist(parsed.data.ticker);
  return NextResponse.json(await list());
}

export async function DELETE(req: Request) {
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await removeFromWatchlist(parsed.data.ticker);
  return NextResponse.json(await list());
}
