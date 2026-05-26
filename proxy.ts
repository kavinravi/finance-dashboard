import { NextResponse, type NextRequest } from "next/server";
import { shouldAllow } from "@/lib/auth/gate";
import { verifySessionToken, SESSION_COOKIE } from "@/lib/auth/session";

export async function proxy(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const appPassword = process.env.APP_PASSWORD;
  const sessionSecret = process.env.SESSION_SECRET;

  const token = req.cookies.get(SESSION_COOKIE)?.value;
  const hasValidSession = sessionSecret ? await verifySessionToken(token, sessionSecret) : false;

  const decision = shouldAllow({
    pathname,
    hasValidSession,
    appPassword,
    sessionSecret,
    onVercel: Boolean(process.env.VERCEL),
  });

  if (decision.allow) return NextResponse.next();

  if (decision.reason === "misconfig") {
    return new NextResponse("App is not configured: APP_PASSWORD/SESSION_SECRET missing.", { status: 503 });
  }

  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = `?next=${encodeURIComponent(pathname)}`;
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|robots.txt).*)"],
};
