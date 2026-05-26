import { NextResponse } from "next/server";
import { pruneExpired } from "@/lib/db/maintenance";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST() {
  try {
    const counts = await pruneExpired();
    return NextResponse.json({ ok: true, ...counts });
  } catch {
    return NextResponse.json({ ok: false, error: "prune_failed" }, { status: 500 });
  }
}
