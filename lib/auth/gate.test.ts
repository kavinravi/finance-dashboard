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
