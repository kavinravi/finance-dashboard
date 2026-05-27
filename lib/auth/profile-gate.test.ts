import { describe, it, expect } from "vitest";
import { needsProfileSelection } from "./profile-gate";

describe("needsProfileSelection", () => {
  it("does not redirect when a profile cookie is present", () => {
    expect(needsProfileSelection("/", true)).toBe(false);
    expect(needsProfileSelection("/ticker/NVDA", true)).toBe(false);
  });
  it("redirects page navigations that have no profile cookie", () => {
    expect(needsProfileSelection("/", false)).toBe(true);
    expect(needsProfileSelection("/watchlist", false)).toBe(true);
  });
  it("never redirects the picker, login, api routes, or assets", () => {
    expect(needsProfileSelection("/select-profile", false)).toBe(false);
    expect(needsProfileSelection("/login", false)).toBe(false);
    expect(needsProfileSelection("/api/watchlist", false)).toBe(false);
    expect(needsProfileSelection("/_next/static/x.js", false)).toBe(false);
    expect(needsProfileSelection("/favicon.ico", false)).toBe(false);
  });
});
