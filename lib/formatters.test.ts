import { describe, it, expect } from "vitest";
import {
  formatPercent,
  formatPrice,
  formatLargeCurrency,
  formatMultiple,
  formatRatioPercent,
} from "./formatters";

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

describe("formatLargeCurrency", () => {
  it("scales to T/B/M and shows N/A for null", () => {
    expect(formatLargeCurrency(3_420_000_000_000)).toBe("$3.42T");
    expect(formatLargeCurrency(1_230_000_000)).toBe("$1.23B");
    expect(formatLargeCurrency(456_700_000)).toBe("$456.70M");
    expect(formatLargeCurrency(-1_230_000_000)).toBe("-$1.23B");
    expect(formatLargeCurrency(null)).toBe("N/A");
  });
});
describe("formatMultiple", () => {
  it("appends the multiplier sign and shows N/A for null", () => {
    expect(formatMultiple(28.41)).toBe("28.41×");
    expect(formatMultiple(null)).toBe("N/A");
  });
});
describe("formatRatioPercent", () => {
  it("renders a fraction as a percent and N/A for null", () => {
    expect(formatRatioPercent(0.243)).toBe("24.3%");
    expect(formatRatioPercent(null)).toBe("N/A");
  });
});
