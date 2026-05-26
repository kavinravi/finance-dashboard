import { describe, it, expect } from "vitest";
import { rsi } from "./rsi";

describe("rsi", () => {
  it("is 100 for a strictly increasing series (no losses)", () => {
    const values = Array.from({ length: 30 }, (_, i) => 10 + i);
    const out = rsi(values, 14);
    expect(out[out.length - 1]).toBeCloseTo(100, 6);
  });
  it("is 0 for a strictly decreasing series (no gains)", () => {
    const values = Array.from({ length: 30 }, (_, i) => 100 - i);
    const out = rsi(values, 14);
    expect(out[out.length - 1]).toBeCloseTo(0, 6);
  });
  it("returns nulls for the warm-up period and matches input length", () => {
    const values = Array.from({ length: 20 }, (_, i) => 50 + (i % 3));
    const out = rsi(values, 14);
    expect(out.length).toBe(values.length);
    expect(out[0]).toBeNull();
    expect(out[13]).toBeNull();
    expect(out[14]).not.toBeNull();
  });
});
