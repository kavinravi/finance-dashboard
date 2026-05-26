import { describe, it, expect, beforeEach } from "vitest";
import { db } from "@/lib/db/client";
import { providerState } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { canCall, recordSuccess, recordError } from "@/lib/db/provider-state";

beforeEach(async () => {
  await db.delete(providerState).where(eq(providerState.provider, "test"));
});

describe("provider-state", () => {
  it("allows calls under the daily limit and blocks at the limit", async () => {
    expect(await canCall("test", 2)).toBe(true);
    await recordSuccess("test", 2);
    expect(await canCall("test", 2)).toBe(true);
    await recordSuccess("test", 2);
    expect(await canCall("test", 2)).toBe(false); // 2/2 used
  });

  it("records errors without incrementing the call count past success calls", async () => {
    await recordError("test", 5, "boom");
    const [row] = await db.select().from(providerState).where(eq(providerState.provider, "test"));
    expect(row.lastError).toBe("boom");
  });
});
