import { NextResponse } from "next/server";
import { z } from "zod";
import { env } from "@/lib/env";
import { createSessionToken, verifyPassword, safeNextPath, SESSION_COOKIE, SESSION_MAX_AGE_MS } from "@/lib/auth/session";

export const runtime = "nodejs";

const bodySchema = z.object({ password: z.string().min(1), next: z.string().optional() });

export async function POST(req: Request) {
  const json = await req.json().catch(() => null);
  const parsed = bodySchema.safeParse(json);
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });

  if (!env.APP_PASSWORD || !env.SESSION_SECRET) {
    return NextResponse.json({ ok: false, error: "not_configured" }, { status: 503 });
  }

  const ok = await verifyPassword(parsed.data.password, env.APP_PASSWORD, env.SESSION_SECRET);
  if (!ok) {
    await new Promise((r) => setTimeout(r, 400)); // small fixed delay to blunt brute force
    return NextResponse.json({ ok: false, error: "invalid" }, { status: 401 });
  }

  const token = await createSessionToken(env.SESSION_SECRET);
  const res = NextResponse.json({ ok: true, next: safeNextPath(parsed.data.next) });
  res.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: Boolean(process.env.VERCEL),
    sameSite: "lax",
    path: "/",
    maxAge: Math.floor(SESSION_MAX_AGE_MS / 1000),
  });
  return res;
}
