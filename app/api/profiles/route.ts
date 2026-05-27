import { NextResponse } from "next/server";
import { z } from "zod";
import { cookies } from "next/headers";
import { listProfiles, createProfile } from "@/lib/db/profiles";
import { PROFILE_COOKIE } from "@/lib/auth/profile-gate";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const createSchema = z.object({ name: z.string().trim().min(1).max(40) });

export async function GET() {
  const all = await listProfiles();
  const jar = await cookies();
  const activeId = jar.get(PROFILE_COOKIE)?.value ?? null;
  return NextResponse.json({ profiles: all.map((p) => ({ id: p.id, name: p.name })), activeId });
}

export async function POST(req: Request) {
  const parsed = createSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  try {
    const profile = await createProfile(parsed.data.name);
    return NextResponse.json({ ok: true, profile: { id: profile.id, name: profile.name } });
  } catch {
    return NextResponse.json({ ok: false, error: "duplicate" }, { status: 409 }); // unique name violation
  }
}
