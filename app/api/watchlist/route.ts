import { NextResponse } from "next/server";
import { z } from "zod";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";
import { getActiveProfileId } from "@/lib/profiles/active";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const schema = z.object({ ticker: z.string().regex(/^[A-Za-z.\-]{1,10}$/) });
const list = async (profileId: string) => ({ tickers: (await getWatchlist(profileId)).map((w) => w.ticker) });

export async function GET() {
  const profileId = await getActiveProfileId();
  if (!profileId) return NextResponse.json({ ok: false, error: "no_profile" }, { status: 409 });
  return NextResponse.json(await list(profileId));
}

export async function POST(req: Request) {
  const profileId = await getActiveProfileId();
  if (!profileId) return NextResponse.json({ ok: false, error: "no_profile" }, { status: 409 });
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await addToWatchlist(profileId, parsed.data.ticker);
  return NextResponse.json(await list(profileId));
}

export async function DELETE(req: Request) {
  const profileId = await getActiveProfileId();
  if (!profileId) return NextResponse.json({ ok: false, error: "no_profile" }, { status: 409 });
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await removeFromWatchlist(profileId, parsed.data.ticker);
  return NextResponse.json(await list(profileId));
}
