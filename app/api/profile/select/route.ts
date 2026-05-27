import { NextResponse } from "next/server";
import { z } from "zod";
import { getProfileById } from "@/lib/db/profiles";
import { safeNextPath } from "@/lib/auth/session";
import { PROFILE_COOKIE } from "@/lib/auth/profile-gate";

export const runtime = "nodejs";

const schema = z.object({ id: z.string().uuid(), next: z.string().optional() });

export async function POST(req: Request) {
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });

  const profile = await getProfileById(parsed.data.id);
  if (!profile) return NextResponse.json({ ok: false, error: "not_found" }, { status: 404 });

  const res = NextResponse.json({ ok: true, next: safeNextPath(parsed.data.next) });
  res.cookies.set(PROFILE_COOKIE, profile.id, {
    httpOnly: true,
    secure: Boolean(process.env.VERCEL),
    sameSite: "lax",
    path: "/",
    // no maxAge → session cookie: clears on browser close so each fresh open re-prompts
  });
  return res;
}
