import { describe, it, expect } from "vitest";
import { env } from "./env";

describe("env (SP2 additions)", () => {
  it("defaults the Gemini model and daily limit", () => {
    expect(env.GEMINI_MODEL).toBe("gemini-3.5-flash");
    expect(env.GEMINI_DAILY_LIMIT).toBeGreaterThan(0);
  });
});
