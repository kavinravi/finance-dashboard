import { describe, it, expect } from "vitest";
import { formatPercent, formatPrice } from "./formatters";

describe("formatters", () => {
  it("formats decimal fractions as signed percentages", () => {
    expect(formatPercent(0.0532)).toBe("+5.32%");
    expect(formatPercent(-0.01)).toBe("-1.00%");
    expect(formatPercent(null)).toBe("—");
  });
  it("formats prices with two decimals", () => {
    expect(formatPrice(262.8)).toBe("262.80");
    expect(formatPrice(null)).toBe("—");
  });
});
