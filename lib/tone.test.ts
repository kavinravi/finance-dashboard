import { describe, it, expect } from "vitest";
import { toneColor, toneLabelText } from "./tone";

describe("toneColor", () => {
  it("maps score ranges to colors", () => {
    expect(toneColor(10)).toBe(toneColor(0));   // bearish band
    expect(toneColor(50)).not.toBe(toneColor(10)); // neutral differs from bearish
    expect(toneColor(90)).not.toBe(toneColor(50)); // bullish differs from neutral
  });
});

describe("toneLabelText", () => {
  it("humanizes labels", () => {
    expect(toneLabelText("somewhat_bullish")).toBe("Somewhat bullish");
    expect(toneLabelText("neutral")).toBe("Neutral");
  });
});
