import { describe, it, expect } from "vitest";
import { findCik } from "./sec";

const MAP = {
  "0": { cik_str: 1045810, ticker: "NVDA", title: "NVIDIA CORP" },
  "1": { cik_str: 320193, ticker: "AAPL", title: "Apple Inc." },
};

describe("findCik", () => {
  it("returns the 10-digit zero-padded CIK for a known ticker", () => {
    expect(findCik(MAP, "AAPL")).toBe("0000320193");
    expect(findCik(MAP, "NVDA")).toBe("0001045810");
  });
  it("is case-insensitive", () => {
    expect(findCik(MAP, "aapl")).toBe("0000320193");
  });
  it("returns null when the ticker is absent", () => {
    expect(findCik(MAP, "ZZZZ")).toBeNull();
  });
});
