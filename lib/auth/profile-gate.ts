export const PROFILE_COOKIE = "fd_profile";

const EXEMPT_EXACT = new Set(["/select-profile", "/login", "/favicon.ico", "/robots.txt"]);

// Once auth has passed, page navigations require an active profile. API routes
// and assets are exempt (the watchlist API enforces a profile itself).
export function needsProfileSelection(pathname: string, hasProfileCookie: boolean): boolean {
  if (hasProfileCookie) return false;
  if (EXEMPT_EXACT.has(pathname)) return false;
  if (pathname.startsWith("/api/")) return false;
  if (pathname.startsWith("/_next/") || pathname.startsWith("/static/")) return false;
  return true;
}
