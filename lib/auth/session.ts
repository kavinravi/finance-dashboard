// Web Crypto HMAC-SHA256 — works in both edge middleware and Node route handlers / Vitest.
const COOKIE_NAME = "fd_session";
const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

export const SESSION_COOKIE = COOKIE_NAME;
export const SESSION_MAX_AGE_MS = THIRTY_DAYS_MS;

function toBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function hmac(message: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  return toBase64Url(new Uint8Array(sig));
}

// Length-checked, branch-uniform compare. Inputs in this module are equal-length
// base64url HMAC digests, so the length check never short-circuits a real compare.
export function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}

export async function createSessionToken(secret: string, now: number = Date.now()): Promise<string> {
  const expiry = String(now + THIRTY_DAYS_MS);
  const sig = await hmac(expiry, secret);
  return `${expiry}.${sig}`;
}

export async function verifySessionToken(
  token: string | undefined,
  secret: string,
  now: number = Date.now(),
): Promise<boolean> {
  if (!token) return false;
  const dot = token.indexOf(".");
  if (dot <= 0) return false;
  const expiryStr = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  const expiry = Number(expiryStr);
  if (!Number.isFinite(expiry) || expiry <= now) return false;
  const expected = await hmac(expiryStr, secret);
  return constantTimeEqual(sig, expected);
}

// Compares HMACs (always equal length) so the raw password length/timing never leaks.
export async function verifyPassword(submitted: string, expected: string, secret: string): Promise<boolean> {
  const [a, b] = await Promise.all([hmac(submitted, secret), hmac(expected, secret)]);
  return constantTimeEqual(a, b);
}

export function safeNextPath(raw: string | null | undefined): string {
  if (!raw) return "/";
  if (!raw.startsWith("/")) return "/";
  if (raw.startsWith("//") || raw.startsWith("/\\")) return "/";
  return raw;
}
