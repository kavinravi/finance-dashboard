import { describe, it, expect } from "vitest";
import { sma } from "./moving-averages";

describe("sma", () => {
  it("returns nulls until the window is full, then simple averages", () => {
    expect(sma([1, 2, 3, 4, 5], 3)).toEqual([null, null, 2, 3, 4]);
  });
  it("returns all nulls when the series is shorter than the window", () => {
    expect(sma([1, 2], 3)).toEqual([null, null]);
  });
});
