# SP4 — Deploy & Harden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app deployable to a private, password-gated `*.vercel.app` URL: add a shared-password gate (middleware + styled login + signed cookie), a provider-health page, opportunistic + manual cache pruning, and security hardening.

**Architecture:** A thin edge `middleware.ts` calls a pure `shouldAllow` decision over an HMAC-signed session cookie; a styled `/login` posts to `/api/login`, which sets the cookie. The gate fails closed on Vercel when secrets are missing, and is off locally when unset. A gated `/health` server page reads `provider_state`; expired cache rows are pruned opportunistically during news fetches and on demand via a `/health` button. Security headers + a disallow-all robots round out hardening. Deploy reuses the existing Neon DB from branch `rebuild-nextjs-mvp`.

**Tech Stack:** Next.js 16 (App Router, edge middleware), TypeScript, Web Crypto (HMAC-SHA256), Drizzle + Neon Postgres, Zod, Vitest, Playwright. Deploy target: Vercel Hobby.

**Spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §9.

**Standing constraints:**
- No AI-authorship traces in commits/docs/branches (no "Claude"/"Anthropic"/"Co-Authored-By"/tool names). Git author `kavinravi` is correct.
- `gemini-3.5-flash` is the verified 2026 stable model — don't let any reviewer "fix" it.
- Do NOT run `pnpm lint`/eslint locally (OOM-crashes); lint is verified in CI. Next 16 does not lint during `build`.
- Tests: `pnpm test` (Vitest; unit + integration that hits live Neon via the `dotenv/config` setup file), `pnpm test:e2e` (Playwright). Type check: `pnpm exec tsc --noEmit`.
- After the build, do a **live smoke with the gate ON** (`pnpm build` + `pnpm start` with `APP_PASSWORD`/`SESSION_SECRET` set) and screenshot the login + health pages before declaring done — mocks can't catch a real redirect/cookie path.
- Deploy itself (Vercel account, env vars, authorizing the push) is performed by the user — see spec §9.9. This plan stops at "locally verified + ready to deploy."

**Verified context (2026-05-26):**
- No `middleware.ts` exists yet; `next.config.ts` is empty; `app/` has no `login`/`health`/`robots`.
- `provider_state` rows exist for fmp/finnhub/gemini/sec (a plain primary-key table; comment is stale but data is correct).
- `articles.expiresAt` is **nullable** (`lt(expiresAt, now)` safely skips nulls); `company_fundamentals.expiresAt` is `notNull`.
- Edge middleware and Node 20 (Vitest) both expose global `crypto.subtle` → one Web-Crypto module works in both runtimes.
- Vitest `setupFiles: ["dotenv/config"]` loads `.env` for all tests; include globs cover `lib/**/*.test.ts`.
- `git remote origin` = `https://github.com/kavinravi/finance-dashboard.git`; branch `rebuild-nextjs-mvp`, clean, 70 commits ahead of `main`.

---

## File Structure

**Create:**
- `lib/auth/session.ts` — Web-Crypto session primitives: `createSessionToken`, `verifySessionToken`, `verifyPassword`, `constantTimeEqual`, `safeNextPath`, `SESSION_COOKIE`, `SESSION_MAX_AGE_MS`.
- `lib/auth/session.test.ts` — unit tests for the above.
- `lib/auth/gate.ts` — pure `shouldAllow(input)` decision.
- `lib/auth/gate.test.ts` — unit tests for `shouldAllow`.
- `middleware.ts` — thin edge wrapper (reads cookie + `process.env`, calls `shouldAllow`).
- `app/login/page.tsx` — server page; wraps the form in `<Suspense>`.
- `components/login-form.tsx` — client login form (`useSearchParams`).
- `app/api/login/route.ts` — POST: verify password, set cookie.
- `app/api/logout/route.ts` — POST: clear cookie.
- `components/app-nav.tsx` — client nav (Home / Health / Log out); hidden on `/login`.
- `lib/db/maintenance.ts` — `pruneExpired()` (articles + fundamentals).
- `lib/db/maintenance.test.ts` — integration (live Neon) for prune + `getAllProviderStates`.
- `app/api/admin/prune/route.ts` — POST (gated): run `pruneExpired`.
- `components/prune-button.tsx` — client "Prune now" button (on `/health`).
- `app/health/page.tsx` — gated server health page.
- `app/robots.ts` — disallow-all robots.
- `tests/e2e/auth.constants.ts` — shared E2E password/secret/state-path constants.
- `tests/e2e/auth.setup.ts` — Playwright setup project: log in once, save `storageState`.
- `tests/e2e/auth.spec.ts` — unauthenticated redirect + wrong/right password flows.

**Modify:**
- `lib/env.ts` — add `APP_PASSWORD`, `SESSION_SECRET` (both optional).
- `.env.example` — document the two new vars.
- `lib/db/provider-state.ts` — add `getAllProviderStates()` + `ProviderStateRow` type.
- `lib/db/articles.ts` — add `pruneExpiredForCompany(companyId)` (and import `lt`).
- `lib/services/news-service.ts` — opportunistic `pruneExpiredForCompany` after upsert.
- `next.config.ts` — security `headers()`.
- `app/layout.tsx` — render `<AppNav/>`.
- `playwright.config.ts` — setup project + `storageState` + gate env on `webServer`.
- `.gitignore` — ignore `tests/e2e/.auth/`.

---

## Task 1: Env vars + `.env.example`

**Files:**
- Modify: `lib/env.ts`
- Modify: `.env.example`

- [ ] **Step 1: Add the two optional vars to the schema + parse object**

In `lib/env.ts`, add to the `z.object({ ... })` schema (after `SEC_USER_AGENT`):

```ts
  APP_PASSWORD: z.string().min(1).optional(),     // SP4 gate password; enforced at runtime by middleware
  SESSION_SECRET: z.string().min(1).optional(),   // SP4 HMAC key for the session cookie
```

And add to the `safeParse({ ... })` object (after `SEC_USER_AGENT`):

```ts
  APP_PASSWORD: process.env.APP_PASSWORD,
  SESSION_SECRET: process.env.SESSION_SECRET,
```

- [ ] **Step 2: Document them in `.env.example`**

Append to `.env.example`:

```bash

# Auth gate (used in SP4) — when both are set the app requires login.
# On Vercel, both MUST be set or the app fails closed. Locally, leave unset for an ungated dev server.
APP_PASSWORD=
SESSION_SECRET=   # a long random string, e.g. `openssl rand -hex 32`
```

- [ ] **Step 3: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add lib/env.ts .env.example
git commit -m "Add APP_PASSWORD and SESSION_SECRET env vars for the auth gate"
```

---

## Task 2: Session primitives (`lib/auth/session.ts`)

**Files:**
- Create: `lib/auth/session.ts`
- Test: `lib/auth/session.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `lib/auth/session.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  createSessionToken, verifySessionToken, verifyPassword, constantTimeEqual, safeNextPath,
} from "./session";

const SECRET = "test-secret-key";

describe("session token", () => {
  it("round-trips a freshly created token", async () => {
    const token = await createSessionToken(SECRET);
    expect(await verifySessionToken(token, SECRET)).toBe(true);
  });

  it("rejects a token signed with a different secret", async () => {
    const token = await createSessionToken(SECRET);
    expect(await verifySessionToken(token, "other-secret")).toBe(false);
  });

  it("rejects a tampered token", async () => {
    const token = await createSessionToken(SECRET);
    const tampered = token.slice(0, -1) + (token.endsWith("A") ? "B" : "A");
    expect(await verifySessionToken(tampered, SECRET)).toBe(false);
  });

  it("rejects an expired token", async () => {
    const token = await createSessionToken(SECRET, 0); // expiry = 0 + 30d → ~1970, already past
    expect(await verifySessionToken(token, SECRET)).toBe(false);
  });

  it("rejects malformed / empty tokens", async () => {
    expect(await verifySessionToken(undefined, SECRET)).toBe(false);
    expect(await verifySessionToken("", SECRET)).toBe(false);
    expect(await verifySessionToken("nodot", SECRET)).toBe(false);
    expect(await verifySessionToken("abc.def", SECRET)).toBe(false);
  });
});

describe("verifyPassword", () => {
  it("accepts the correct password and rejects wrong ones", async () => {
    expect(await verifyPassword("hunter2", "hunter2", SECRET)).toBe(true);
    expect(await verifyPassword("nope", "hunter2", SECRET)).toBe(false);
  });
});

describe("constantTimeEqual", () => {
  it("compares equal-length strings", () => {
    expect(constantTimeEqual("abc", "abc")).toBe(true);
    expect(constantTimeEqual("abc", "abd")).toBe(false);
    expect(constantTimeEqual("abc", "abcd")).toBe(false);
  });
});

describe("safeNextPath", () => {
  it("allows same-origin absolute paths", () => {
    expect(safeNextPath("/ticker/AAPL")).toBe("/ticker/AAPL");
    expect(safeNextPath("/health")).toBe("/health");
  });
  it("rejects protocol-relative, backslash, absolute-URL, and junk", () => {
    expect(safeNextPath("//evil.com")).toBe("/");
    expect(safeNextPath("/\\evil.com")).toBe("/");
    expect(safeNextPath("https://evil.com")).toBe("/");
    expect(safeNextPath("ticker")).toBe("/");
    expect(safeNextPath(null)).toBe("/");
    expect(safeNextPath(undefined)).toBe("/");
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm exec vitest run lib/auth/session.test.ts`
Expected: FAIL — `Cannot find module './session'`.

- [ ] **Step 3: Implement `lib/auth/session.ts`**

```ts
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm exec vitest run lib/auth/session.test.ts`
Expected: PASS (all assertions).

- [ ] **Step 5: Commit**

```bash
git add lib/auth/session.ts lib/auth/session.test.ts
git commit -m "Add Web Crypto session primitives for the auth gate"
```

---

## Task 3: Gate decision (`lib/auth/gate.ts`)

**Files:**
- Create: `lib/auth/gate.ts`
- Test: `lib/auth/gate.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `lib/auth/gate.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { shouldAllow, type GateInput } from "./gate";

function base(overrides: Partial<GateInput> = {}): GateInput {
  return {
    pathname: "/ticker/AAPL",
    hasValidSession: false,
    appPassword: "pw",
    sessionSecret: "secret",
    onVercel: false,
    ...overrides,
  };
}

describe("shouldAllow", () => {
  it("always allows the login route, auth APIs, and static assets", () => {
    for (const pathname of ["/login", "/api/login", "/api/logout", "/_next/abc", "/favicon.ico", "/robots.txt"]) {
      expect(shouldAllow(base({ pathname, hasValidSession: false }))).toEqual({ allow: true });
    }
  });

  it("allows a gated path when the session is valid", () => {
    expect(shouldAllow(base({ hasValidSession: true }))).toEqual({ allow: true });
  });

  it("redirects to login on a gated path without a session", () => {
    expect(shouldAllow(base({ hasValidSession: false }))).toEqual({ allow: false, reason: "login" });
  });

  it("fails closed on Vercel when not configured", () => {
    expect(shouldAllow(base({ appPassword: undefined, onVercel: true })))
      .toEqual({ allow: false, reason: "misconfig" });
    expect(shouldAllow(base({ sessionSecret: undefined, onVercel: true })))
      .toEqual({ allow: false, reason: "misconfig" });
  });

  it("is open locally when not configured (gate off)", () => {
    expect(shouldAllow(base({ appPassword: undefined, onVercel: false }))).toEqual({ allow: true });
    expect(shouldAllow(base({ sessionSecret: undefined, onVercel: false }))).toEqual({ allow: true });
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm exec vitest run lib/auth/gate.test.ts`
Expected: FAIL — `Cannot find module './gate'`.

- [ ] **Step 3: Implement `lib/auth/gate.ts`**

```ts
const ALWAYS_ALLOW_EXACT = new Set(["/login", "/api/login", "/api/logout", "/favicon.ico", "/robots.txt"]);

export type GateInput = {
  pathname: string;
  hasValidSession: boolean;
  appPassword: string | undefined;
  sessionSecret: string | undefined;
  onVercel: boolean;
};

export type GateDecision = { allow: true } | { allow: false; reason: "login" | "misconfig" };

export function shouldAllow(input: GateInput): GateDecision {
  const { pathname, hasValidSession, appPassword, sessionSecret, onVercel } = input;

  if (ALWAYS_ALLOW_EXACT.has(pathname) || pathname.startsWith("/_next/") || pathname.startsWith("/static/")) {
    return { allow: true };
  }

  const configured = Boolean(appPassword) && Boolean(sessionSecret);
  if (!configured) {
    return onVercel ? { allow: false, reason: "misconfig" } : { allow: true };
  }

  return hasValidSession ? { allow: true } : { allow: false, reason: "login" };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm exec vitest run lib/auth/gate.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/auth/gate.ts lib/auth/gate.test.ts
git commit -m "Add pure gate decision for the auth middleware"
```

---

## Task 4: Edge middleware (`middleware.ts`)

**Files:**
- Create: `middleware.ts`

Middleware reads `process.env` directly (not `@/lib/env`) so the edge bundle stays minimal and never triggers full env validation; it only needs the two gate vars + `VERCEL`.

- [ ] **Step 1: Implement `middleware.ts`**

```ts
import { NextResponse, type NextRequest } from "next/server";
import { shouldAllow } from "@/lib/auth/gate";
import { verifySessionToken, SESSION_COOKIE } from "@/lib/auth/session";

export async function middleware(req: NextRequest) {
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
```

- [ ] **Step 2: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add middleware.ts
git commit -m "Add edge middleware enforcing the password gate"
```

> Behavioral verification (redirect / cookie / fail-closed) is covered by the E2E suite in Task 9 and the live smoke in Task 10 — middleware can't be meaningfully unit-tested in this setup, so its logic lives in the unit-tested `gate.ts` + `session.ts`.

---

## Task 5: Login + logout (routes + page + form)

**Files:**
- Create: `app/api/login/route.ts`
- Create: `app/api/logout/route.ts`
- Create: `components/login-form.tsx`
- Create: `app/login/page.tsx`

- [ ] **Step 1: Implement `app/api/login/route.ts`**

```ts
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
```

- [ ] **Step 2: Implement `app/api/logout/route.ts`**

```ts
import { NextResponse } from "next/server";
import { SESSION_COOKIE } from "@/lib/auth/session";

export const runtime = "nodejs";

export async function POST() {
  const res = NextResponse.json({ ok: true });
  res.cookies.set(SESSION_COOKIE, "", { httpOnly: true, path: "/", maxAge: 0 });
  return res;
}
```

- [ ] **Step 3: Implement `components/login-form.tsx`**

```tsx
"use client";
import { useState } from "react";
import { useSearchParams } from "next/navigation";

export function LoginForm() {
  const params = useSearchParams();
  const next = params.get("next") ?? "/";
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password, next }),
    });
    if (res.ok) {
      const data = await res.json();
      window.location.href = typeof data.next === "string" ? data.next : "/";
      return;
    }
    setLoading(false);
    setError(res.status === 503 ? "App is not configured." : "Incorrect password.");
  }

  return (
    <form onSubmit={onSubmit} className="w-full max-w-sm space-y-4">
      <div className="space-y-1">
        <label htmlFor="password" className="block text-sm">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="w-full rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm"
        />
      </div>
      {error && <p className="text-sm text-red-400">{error}</p>}
      <button
        type="submit"
        disabled={loading}
        className="w-full rounded bg-blue-600 px-3 py-2 text-sm font-medium disabled:opacity-50"
      >
        Sign in
      </button>
    </form>
  );
}
```

- [ ] **Step 4: Implement `app/login/page.tsx`**

`useSearchParams` requires a Suspense boundary in Next 16, so the server page wraps the client form.

```tsx
import { Suspense } from "react";
import { LoginForm } from "@/components/login-form";

export const metadata = { title: "Sign in · Finance Dashboard" };

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1 text-center">
          <h1 className="text-xl font-semibold">Finance Dashboard</h1>
          <p className="text-sm text-neutral-400">Enter the password to continue.</p>
        </div>
        <Suspense>
          <LoginForm />
        </Suspense>
      </div>
    </main>
  );
}
```

- [ ] **Step 5: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/api/login/route.ts app/api/logout/route.ts components/login-form.tsx app/login/page.tsx
git commit -m "Add login/logout routes and styled login page"
```

---

## Task 6: App nav + layout wiring

**Files:**
- Create: `components/app-nav.tsx`
- Modify: `app/layout.tsx`

- [ ] **Step 1: Implement `components/app-nav.tsx`**

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function AppNav() {
  const pathname = usePathname();
  if (pathname === "/login") return null;

  async function logout() {
    await fetch("/api/logout", { method: "POST" });
    window.location.href = "/login";
  }

  return (
    <nav className="flex items-center justify-end gap-4 border-b border-neutral-800 px-4 py-2 text-sm">
      <Link href="/" className="text-neutral-300 hover:text-white">Home</Link>
      <Link href="/health" className="text-neutral-300 hover:text-white">Health</Link>
      <button onClick={logout} className="text-neutral-300 hover:text-white">Log out</button>
    </nav>
  );
}
```

- [ ] **Step 2: Render `<AppNav/>` in `app/layout.tsx`**

Replace the body of `app/layout.tsx` with:

```tsx
import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { AppNav } from "@/components/app-nav";

export const metadata: Metadata = { title: "Finance Dashboard", description: "Investing research" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        <Providers>
          <AppNav />
          {children}
        </Providers>
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add components/app-nav.tsx app/layout.tsx
git commit -m "Add app nav with logout and health links"
```

---

## Task 7: Cache pruning + provider-state read

**Files:**
- Create: `lib/db/maintenance.ts`
- Test: `lib/db/maintenance.test.ts`
- Modify: `lib/db/articles.ts`
- Modify: `lib/db/provider-state.ts`
- Modify: `lib/services/news-service.ts`
- Create: `app/api/admin/prune/route.ts`
- Create: `components/prune-button.tsx`

- [ ] **Step 1: Write the failing integration test**

Create `lib/db/maintenance.test.ts` (hits live Neon, like the other integration tests; seeds a throwaway company and cleans up):

```ts
import { describe, it, expect, afterAll } from "vitest";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { companies, articles, companyFundamentals } from "@/lib/db/schema";
import { pruneExpired } from "@/lib/db/maintenance";
import { pruneExpiredForCompany } from "@/lib/db/articles";
import { getAllProviderStates } from "@/lib/db/provider-state";

const TICKER = `ZZPRUNE${Date.now()}`;
const past = new Date(Date.now() - 86_400_000);
const future = new Date(Date.now() + 86_400_000);
let companyId = "";

afterAll(async () => {
  if (companyId) {
    await db.delete(articles).where(eq(articles.companyId, companyId));
    await db.delete(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
    await db.delete(companies).where(eq(companies.id, companyId));
  }
});

describe("pruning (integration, live Neon)", () => {
  it("pruneExpired deletes expired articles + fundamentals and keeps fresh ones", async () => {
    const [c] = await db.insert(companies).values({ ticker: TICKER, name: "Prune Test Co" }).returning({ id: companies.id });
    companyId = c.id;

    await db.insert(articles).values([
      { companyId, source: "finnhub", url: "https://ex.com/expired", urlHash: `${TICKER}-exp`, title: "Expired", publishedAt: past, expiresAt: past },
      { companyId, source: "finnhub", url: "https://ex.com/fresh", urlHash: `${TICKER}-fresh`, title: "Fresh", publishedAt: past, expiresAt: future },
    ]);
    await db.insert(companyFundamentals).values([
      { companyId, conceptsJson: {}, source: "sec_edgar", expiresAt: past },
    ]);

    const counts = await pruneExpired();
    expect(counts.articles).toBeGreaterThanOrEqual(1);
    expect(counts.fundamentals).toBeGreaterThanOrEqual(1);

    const remaining = await db.select({ urlHash: articles.urlHash }).from(articles).where(eq(articles.companyId, companyId));
    expect(remaining.map((r) => r.urlHash)).toEqual([`${TICKER}-fresh`]);

    const fund = await db.select({ id: companyFundamentals.id }).from(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
    expect(fund).toHaveLength(0);
  });

  it("pruneExpiredForCompany removes only that company's expired rows", async () => {
    await db.insert(articles).values([
      { companyId, source: "finnhub", url: "https://ex.com/expired2", urlHash: `${TICKER}-exp2`, title: "Expired2", publishedAt: past, expiresAt: past },
    ]);
    const n = await pruneExpiredForCompany(companyId);
    expect(n).toBe(1);
    const remaining = await db.select({ urlHash: articles.urlHash }).from(articles).where(eq(articles.companyId, companyId));
    expect(remaining.map((r) => r.urlHash)).toEqual([`${TICKER}-fresh`]);
  });

  it("getAllProviderStates returns rows whose provider is a string", async () => {
    const rows = await getAllProviderStates();
    expect(Array.isArray(rows)).toBe(true);
    for (const r of rows) expect(typeof r.provider).toBe("string");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm exec vitest run lib/db/maintenance.test.ts`
Expected: FAIL — `Cannot find module '@/lib/db/maintenance'` (and `pruneExpiredForCompany` / `getAllProviderStates` not exported).

- [ ] **Step 3: Implement `lib/db/maintenance.ts`**

`.returning({ id })` is used so `.length` is a reliable deleted-row count (a bare Neon `delete` returns an empty array).

```ts
import { db } from "./client";
import { articles, companyFundamentals } from "./schema";
import { lt } from "drizzle-orm";

export async function pruneExpired(now: Date = new Date()): Promise<{ articles: number; fundamentals: number }> {
  const a = await db.delete(articles).where(lt(articles.expiresAt, now)).returning({ id: articles.id });
  const f = await db.delete(companyFundamentals).where(lt(companyFundamentals.expiresAt, now)).returning({ id: companyFundamentals.id });
  return { articles: a.length, fundamentals: f.length };
}
```

- [ ] **Step 4: Add `pruneExpiredForCompany` to `lib/db/articles.ts`**

Change the import line to add `lt`:

```ts
import { and, eq, gte, gt, desc, inArray, lt } from "drizzle-orm";
```

Append:

```ts
export async function pruneExpiredForCompany(companyId: string, now: Date = new Date()): Promise<number> {
  const r = await db.delete(articles)
    .where(and(eq(articles.companyId, companyId), lt(articles.expiresAt, now)))
    .returning({ id: articles.id });
  return r.length;
}
```

- [ ] **Step 5: Add `getAllProviderStates` to `lib/db/provider-state.ts`**

Append (the file already imports `db` and `providerState`):

```ts
export type ProviderStateRow = typeof providerState.$inferSelect;

export async function getAllProviderStates(): Promise<ProviderStateRow[]> {
  return db.select().from(providerState).orderBy(providerState.provider);
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pnpm exec vitest run lib/db/maintenance.test.ts`
Expected: PASS.

- [ ] **Step 7: Wire opportunistic pruning into `lib/services/news-service.ts`**

Change the import on line 2 to add `pruneExpiredForCompany`:

```ts
import { upsertArticles, getRecentArticles, newestArticleCreatedAt, pruneExpiredForCompany, type ArticleRow } from "@/lib/db/articles";
```

Inside the `if (opts.force || !fresh) { ... }` block, immediately after `await upsertArticles(company.id, dedupeArticles([...fh, ...yr]));`, add:

```ts
    await pruneExpiredForCompany(company.id).catch(() => 0); // best-effort; never break the news path
```

- [ ] **Step 8: Implement `app/api/admin/prune/route.ts`** (gated by middleware — not in the allow-list)

```ts
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
```

- [ ] **Step 9: Implement `components/prune-button.tsx`**

```tsx
"use client";
import { useState } from "react";

export function PruneButton() {
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onClick() {
    setLoading(true);
    setMsg(null);
    const res = await fetch("/api/admin/prune", { method: "POST" });
    setLoading(false);
    if (res.ok) {
      const d = await res.json();
      setMsg(`Removed ${d.articles} articles, ${d.fundamentals} snapshots.`);
    } else {
      setMsg("Prune failed.");
    }
  }

  return (
    <div className="space-y-2">
      <button
        onClick={onClick}
        disabled={loading}
        className="rounded bg-neutral-800 px-3 py-1.5 text-sm hover:bg-neutral-700 disabled:opacity-50"
      >
        {loading ? "Pruning…" : "Prune now"}
      </button>
      {msg && <p className="text-sm text-neutral-400">{msg}</p>}
    </div>
  );
}
```

- [ ] **Step 10: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add lib/db/maintenance.ts lib/db/maintenance.test.ts lib/db/articles.ts lib/db/provider-state.ts lib/services/news-service.ts app/api/admin/prune/route.ts components/prune-button.tsx
git commit -m "Add cache pruning (opportunistic + manual) and provider-state read"
```

---

## Task 8: Health page + robots + security headers

**Files:**
- Create: `app/health/page.tsx`
- Create: `app/robots.ts`
- Modify: `next.config.ts`

- [ ] **Step 1: Implement `app/health/page.tsx`** (gated; reads provider state + config, hosts the prune button)

```tsx
import { getAllProviderStates } from "@/lib/db/provider-state";
import { env } from "@/lib/env";
import { PruneButton } from "@/components/prune-button";

export const dynamic = "force-dynamic";

function fmt(d: Date | null): string {
  return d ? new Date(d).toISOString().replace("T", " ").slice(0, 19) + "Z" : "—";
}

export default async function HealthPage() {
  let rows: Awaited<ReturnType<typeof getAllProviderStates>> = [];
  let dbOk = true;
  try {
    rows = await getAllProviderStates();
  } catch {
    dbOk = false;
  }

  const configured: Record<string, boolean> = {
    FINNHUB_API_KEY: Boolean(env.FINNHUB_API_KEY),
    GEMINI_API_KEY: Boolean(env.GEMINI_API_KEY),
    SEC_USER_AGENT: Boolean(env.SEC_USER_AGENT),
    APP_PASSWORD: Boolean(env.APP_PASSWORD),
  };

  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">System health</h1>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Configuration</h2>
        <p className="text-sm">Database: {dbOk ? "reachable" : "unreachable"}</p>
        <p className="text-sm">Server time: {fmt(new Date())}</p>
        <ul className="text-sm text-neutral-300">
          {Object.entries(configured).map(([k, v]) => (
            <li key={k}>{k}: {v ? "set" : "missing"}</li>
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Providers</h2>
        {dbOk ? (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-neutral-800 text-left">
                <th className="py-1 pr-4">Provider</th>
                <th className="py-1 pr-4">Calls today</th>
                <th className="py-1 pr-4">Limit</th>
                <th className="py-1 pr-4">Last success</th>
                <th className="py-1 pr-4">Last error</th>
                <th className="py-1 pr-4">Error msg</th>
                <th className="py-1 pr-4">Resets</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.provider} className="border-b border-neutral-900">
                  <td className="py-1 pr-4">{r.provider}</td>
                  <td className="py-1 pr-4">{r.callsToday}</td>
                  <td className="py-1 pr-4">{r.dailyLimit}</td>
                  <td className="py-1 pr-4">{fmt(r.lastSuccessAt)}</td>
                  <td className="py-1 pr-4">{fmt(r.lastErrorAt)}</td>
                  <td className="max-w-[16rem] truncate py-1 pr-4">{r.lastError ?? "—"}</td>
                  <td className="py-1 pr-4">{fmt(r.resetAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-red-400">Database unreachable — cannot read provider state.</p>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Maintenance</h2>
        <PruneButton />
      </section>
    </main>
  );
}
```

- [ ] **Step 2: Implement `app/robots.ts`**

```ts
import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return { rules: { userAgent: "*", disallow: "/" } };
}
```

- [ ] **Step 3: Add security headers in `next.config.ts`**

```ts
import type { NextConfig } from "next";

const securityHeaders = [
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains; preload" },
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
```

- [ ] **Step 4: Type-check**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add app/health/page.tsx app/robots.ts next.config.ts
git commit -m "Add health page, disallow-all robots, and security headers"
```

---

## Task 9: E2E auth suite (Playwright)

**Files:**
- Create: `tests/e2e/auth.constants.ts`
- Create: `tests/e2e/auth.setup.ts`
- Create: `tests/e2e/auth.spec.ts`
- Modify: `playwright.config.ts`
- Modify: `.gitignore`

- [ ] **Step 1: Create `tests/e2e/auth.constants.ts`**

```ts
export const E2E_PASSWORD = "e2e-test-password";
export const E2E_SESSION_SECRET = "e2e-test-session-secret-not-for-prod";
export const E2E_STATE_PATH = "tests/e2e/.auth/state.json";
```

- [ ] **Step 2: Rewrite `playwright.config.ts`** (gate ON via webServer env; setup project saves storageState; chromium reuses it)

```ts
import { defineConfig, devices } from "@playwright/test";
import { E2E_PASSWORD, E2E_SESSION_SECRET, E2E_STATE_PATH } from "./tests/e2e/auth.constants";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:3000" },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], storageState: E2E_STATE_PATH },
      dependencies: ["setup"],
      testIgnore: /auth\.setup\.ts/,
    },
  ],
  // reuseExistingServer is false so the server always carries the gate env below.
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:3000",
    reuseExistingServer: false,
    timeout: 120_000,
    env: { APP_PASSWORD: E2E_PASSWORD, SESSION_SECRET: E2E_SESSION_SECRET },
  },
});
```

- [ ] **Step 3: Create `tests/e2e/auth.setup.ts`** (logs in once, persists the cookie)

```ts
import { test as setup, expect } from "@playwright/test";
import fs from "node:fs";
import { E2E_PASSWORD, E2E_STATE_PATH } from "./auth.constants";

setup("authenticate", async ({ request }) => {
  fs.mkdirSync("tests/e2e/.auth", { recursive: true });
  const res = await request.post("/api/login", { data: { password: E2E_PASSWORD, next: "/" } });
  expect(res.ok()).toBeTruthy();
  await request.storageState({ path: E2E_STATE_PATH });
});
```

- [ ] **Step 4: Create `tests/e2e/auth.spec.ts`** (runs unauthenticated by clearing storageState)

```ts
import { test, expect } from "@playwright/test";
import { E2E_PASSWORD } from "./auth.constants";

test.use({ storageState: { cookies: [], origins: [] } });

test("unauthenticated request is redirected to /login", async ({ page }) => {
  await page.goto("/ticker/NVDA");
  await expect(page).toHaveURL(/\/login/);
  await expect(page.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("wrong password shows an error and stays on /login", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/password/i).fill("definitely-wrong");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText(/incorrect password/i)).toBeVisible();
  await expect(page).toHaveURL(/\/login/);
});

test("correct password signs in and lands on the homepage", async ({ page }) => {
  await page.goto("/login?next=%2F");
  await page.getByLabel(/password/i).fill(E2E_PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL("http://localhost:3000/");
  await expect(page.getByPlaceholder(/Search ticker/i)).toBeVisible();
});
```

- [ ] **Step 5: Ignore the saved auth state**

Append to `.gitignore`:

```
tests/e2e/.auth/
```

- [ ] **Step 6: Run the E2E suite**

Run: `pnpm test:e2e`
Expected: the `setup` project authenticates, then all `chromium` specs pass — the existing smoke specs (now behind the gate) reuse the saved session, and the three `auth.spec.ts` tests pass. (Ensure nothing else is already serving `http://localhost:3000`, since `reuseExistingServer` is false.)

- [ ] **Step 7: Commit**

```bash
git add tests/e2e/auth.constants.ts tests/e2e/auth.setup.ts tests/e2e/auth.spec.ts playwright.config.ts .gitignore
git commit -m "Add E2E auth suite and run existing smokes behind the gate"
```

---

## Task 10: Full verification + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Full unit + integration suite**

Run: `pnpm test`
Expected: all SP1–SP4 Vitest suites pass (no regressions; new `session`, `gate`, `maintenance` suites green).

- [ ] **Step 2: Type-check the whole project**

Run: `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Production build**

Run: `pnpm build`
Expected: build succeeds (Next 16 does not lint during build; lint is a CI concern).

- [ ] **Step 4: Live smoke with the gate ON**

Start a production server with the gate configured (use a throwaway password + a generated secret):

```bash
SECRET=$(openssl rand -hex 32)
APP_PASSWORD=localtest SESSION_SECRET="$SECRET" pnpm start
```

In a second shell, verify (do NOT echo real secrets):

```bash
# Redirect to /login when unauthenticated (expect 307 + location: /login)
curl -sS -o /dev/null -D - "http://localhost:3000/ticker/NVDA" | grep -iE "HTTP/|location"

# Security headers present on a normal response
curl -sS -o /dev/null -D - "http://localhost:3000/login" | grep -iE "x-frame-options|x-content-type-options|referrer-policy|strict-transport-security"

# robots.txt disallows
curl -sS "http://localhost:3000/robots.txt"

# Wrong password rejected (expect HTTP 401)
curl -sS -o /dev/null -w "%{http_code}\n" -X POST "http://localhost:3000/api/login" -H "content-type: application/json" -d '{"password":"wrong"}'

# Correct password accepted (expect 200 + a set-cookie: fd_session=...)
curl -sS -D - -o /dev/null -X POST "http://localhost:3000/api/login" -H "content-type: application/json" -d '{"password":"localtest"}' | grep -iE "HTTP/|set-cookie"
```

Expected: gated route 307→`/login`; all four security headers present; robots disallows `/`; wrong password → 401; correct password → 200 + `Set-Cookie: fd_session=`.

- [ ] **Step 5: Visual confirmation (screenshots)**

With the gated server still running, capture the login page and (after authenticating in a browser) the `/health` page. If the preview MCP is unreachable from Bash (as in prior SPs), use a throwaway Playwright screenshot script against `http://localhost:3000/login` and `/health`. Confirm: login renders centered with a password field + "Sign in"; `/health` shows the provider table (fmp/finnhub/gemini/sec), the configuration block, and the "Prune now" button.

- [ ] **Step 6: Stop the server**

Stop the `pnpm start` process (Ctrl-C / kill). An exit code of 143 (SIGTERM) is expected and not a failure.

- [ ] **Step 7: Final commit (if any verification-driven fixes were made)**

```bash
git add -A
git commit -m "SP4 verification fixes"
```

(If no fixes were needed, skip — don't create an empty commit.)

---

## Self-Review (completed during planning)

**1. Spec coverage (§9):**
- §9.2 env (`APP_PASSWORD`/`SESSION_SECRET`) → Task 1. ✅
- §9.3 gate (`session.ts`, `gate.ts`, `middleware.ts`, login/logout routes + page, logout control) → Tasks 2–6. ✅
- §9.4 health page + `getAllProviderStates` → Tasks 7 (read) + 8 (page). ✅
- §9.5 pruning (`pruneExpired`, `pruneExpiredForCompany`, opportunistic news-service hook, admin route, button) → Task 7. ✅
- §9.6 hardening (headers, robots) → Task 8. ✅
- §9.7 error handling (fail-closed, 401, gated admin/health, DB-unreachable state) → Tasks 4/5/7/8. ✅
- §9.8 testing (session/gate/safeNextPath units; prune/getAll integration; E2E redirect+wrong+right; live smoke) → Tasks 2/3/7/9/10. ✅
- §9.9 deploy runbook → user steps, intentionally out of code scope (noted in header). ✅
- §9.10 acceptance → exercised by Tasks 9 + 10.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✅

**3. Type consistency:** `SESSION_COOKIE`/`SESSION_MAX_AGE_MS`/`verifyPassword`/`createSessionToken`/`verifySessionToken`/`safeNextPath` (Task 2) are used consistently in Tasks 4–5. `shouldAllow`/`GateInput`/`GateDecision` (Task 3) match Task 4 usage. `pruneExpired` returns `{ articles, fundamentals }` (Task 7) — consumed identically by the admin route + button. `getAllProviderStates`/`ProviderStateRow` (Task 7) match the health page (Task 8). `pruneExpiredForCompany` (Task 7) imported by news-service. ✅

---

## Execution Handoff

After the plan is reviewed, execute task-by-task. Recommended: subagent-driven development (fresh subagent per task + spec/quality review between tasks), with the controller personally performing the Task 10 live smoke (mocks can't catch a real redirect/cookie path). Tasks 2, 3, and 7 are TDD with real failing-first tests; Tasks 4–6 and 8 are write-then-typecheck (behavior verified by the Task 9 E2E + Task 10 live smoke).
