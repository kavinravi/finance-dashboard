import { describe, it, expect } from "vitest";
import { ema, macd } from "./macd";

describe("ema", () => {
  it("equals the constant for a constant series", () => {
    const out = ema([5, 5, 5, 5, 5], 3);
    expect(out[out.length - 1]).toBeCloseTo(5, 10);
  });
});

describe("macd", () => {
  it("yields ~0 macd and signal for a constant series", () => {
    const values = Array.from({ length: 60 }, () => 42);
    const { macdLine, signalLine, histogram } = macd(values);
    const n = values.length - 1;
    expect(macdLine[n]).toBeCloseTo(0, 6);
    expect(signalLine[n]).toBeCloseTo(0, 6);
    expect(histogram[n]).toBeCloseTo(0, 6);
  });
  it("matches input length for each line", () => {
    const values = Array.from({ length: 60 }, (_, i) => 10 + Math.sin(i));
    const { macdLine, signalLine, histogram } = macd(values);
    expect(macdLine.length).toBe(values.length);
    expect(signalLine.length).toBe(values.length);
    expect(histogram.length).toBe(values.length);
  });
});
