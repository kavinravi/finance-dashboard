import { NextResponse } from "next/server";
import { z } from "zod";
import { renameProfile, deleteProfile } from "@/lib/db/profiles";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const renameSchema = z.object({ name: z.string().trim().min(1).max(40) });

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const parsed = renameSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  try {
    await renameProfile(id, parsed.data.name);
    return NextResponse.json({ ok: true });
  } catch {
    return NextResponse.json({ ok: false, error: "duplicate" }, { status: 409 });
  }
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  await deleteProfile(id); // watchlist rows cascade
  return NextResponse.json({ ok: true });
}
