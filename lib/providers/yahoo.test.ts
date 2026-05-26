import { describe, it, expect, vi } from "vitest";

const { chart, search } = vi.hoisted(() => ({
  chart: vi.fn(),
  search: vi.fn(),
}));
vi.mock("yahoo-finance2", () => ({
  default: class { chart = chart; search = search; },
}));

import { yahoo } from "./yahoo";

describe("yahoo provider", () => {
  it("maps chart() quotes to PriceBars (ascending, adjclose lowercase)", async () => {
    chart.mockResolvedValue({
      quotes: [
        { date: new Date("2025-05-01T00:00:00Z"), open: 1, high: 2, low: 0.5, close: 1.5, volume: 50, adjclose: 1.4 },
        { date: new Date("2025-05-02T00:00:00Z"), open: 2, high: 3, low: 1, close: 2.5, volume: 100, adjclose: 2.4 },
      ],
    });
    const bars = await yahoo.dailyPrices("AAPL", "2025-05-01", "2025-05-03");
    expect(bars).toHaveLength(2);
    expect(bars[0]).toEqual({ date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, adjClose: 1.4, volume: 50 });
  });

  it("maps search() quotes to SearchResults", async () => {
    search.mockResolvedValue({
      quotes: [{ symbol: "NVDA", exchange: "NMS", shortname: "NVIDIA Corp", typeDisp: "Equity" }],
      news: [],
    });
    const res = await yahoo.search("nvidia");
    expect(res[0]).toEqual({ symbol: "NVDA", name: "NVIDIA Corp", exchange: "NMS", assetType: "stock", source: "yahoo" });
  });
});
