import { describe, it, expect } from "vitest";
import { toneColor, toneLabelText } from "./tone";

describe("toneColor", () => {
  it("maps score ranges to colors", () => {
    expect(toneColor(10)).toBe(toneColor(0));   // bearish band
    expect(toneColor(50)).not.toBe(toneColor(10)); // neutral differs from bearish
    expect(toneColor(90)).not.toBe(toneColor(50)); // bullish differs from neutral
  });

  it("returns the exact band color at boundaries", () => {
    expect(toneColor(30)).toBe("#ef4444");
    expect(toneColor(31)).toBe("#f59e0b");
    expect(toneColor(45)).toBe("#f59e0b");
    expect(toneColor(55)).toBe("#a3a3a3");
    expect(toneColor(70)).toBe("#84cc16");
    expect(toneColor(71)).toBe("#22c55e");
  });
});

describe("toneLabelText", () => {
  it("humanizes labels", () => {
    expect(toneLabelText("bearish")).toBe("Bearish");
    expect(toneLabelText("somewhat_bearish")).toBe("Somewhat bearish");
    expect(toneLabelText("neutral")).toBe("Neutral");
    expect(toneLabelText("somewhat_bullish")).toBe("Somewhat bullish");
    expect(toneLabelText("bullish")).toBe("Bullish");
  });
});
