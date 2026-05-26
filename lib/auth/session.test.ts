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
