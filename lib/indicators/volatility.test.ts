import { describe, it, expect } from "vitest";
import { rollingVolatility } from "./volatility";

describe("rollingVolatility", () => {
  it("is 0 for constant prices (zero daily returns)", () => {
    const closes = Array.from({ length: 10 }, () => 100);
    const out = rollingVolatility(closes, 5);
    expect(out[out.length - 1]).toBeCloseTo(0, 10);
  });
  it("returns null until enough returns exist", () => {
    const closes = [100, 101, 102];
    const out = rollingVolatility(closes, 5);
    expect(out[0]).toBeNull();
    expect(out[2]).toBeNull();
  });
});
