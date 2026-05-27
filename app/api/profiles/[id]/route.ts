import { NextResponse } from "next/server";
import { z } from "zod";
import { listProfiles, renameProfile, deleteProfile } from "@/lib/db/profiles";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const idSchema = z.string().uuid();
const renameSchema = z.object({ name: z.string().trim().min(1).max(40) });

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!idSchema.safeParse(id).success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  const parsed = renameSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  try {
    await renameProfile(id, parsed.data.name);
    return NextResponse.json({ ok: true });
  } catch (err) {
    const taken = (await listProfiles()).some((p) => p.name === parsed.data.name && p.id !== id);
    if (taken) return NextResponse.json({ ok: false, error: "duplicate" }, { status: 409 });
    throw err;
  }
}

export async function DELETE(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!idSchema.safeParse(id).success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await deleteProfile(id); // watchlist rows cascade
  return NextResponse.json({ ok: true });
}
