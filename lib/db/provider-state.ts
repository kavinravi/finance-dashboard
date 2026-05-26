import { db } from "./client";
import { providerState } from "./schema";
import { eq } from "drizzle-orm";

function nextResetAt(): Date {
  const d = new Date();
  d.setUTCHours(24, 0, 0, 0); // next UTC midnight
  return d;
}

async function ensureRow(provider: string, dailyLimit: number) {
  const [row] = await db.select().from(providerState).where(eq(providerState.provider, provider));
  if (!row) {
    await db.insert(providerState).values({ provider, dailyLimit, callsToday: 0, resetAt: nextResetAt() })
      .onConflictDoNothing();
    return (await db.select().from(providerState).where(eq(providerState.provider, provider)))[0];
  }
  if (row.resetAt.getTime() <= Date.now()) {
    await db.update(providerState).set({ callsToday: 0, resetAt: nextResetAt() })
      .where(eq(providerState.provider, provider));
    return { ...row, callsToday: 0, resetAt: nextResetAt() };
  }
  return row;
}

export async function canCall(provider: string, dailyLimit: number): Promise<boolean> {
  const row = await ensureRow(provider, dailyLimit);
  return row.callsToday < dailyLimit;
}

export async function recordSuccess(provider: string, dailyLimit: number): Promise<void> {
  const row = await ensureRow(provider, dailyLimit);
  await db.update(providerState)
    .set({ callsToday: row.callsToday + 1, lastSuccessAt: new Date() })
    .where(eq(providerState.provider, provider));
}

export async function recordError(provider: string, dailyLimit: number, message: string): Promise<void> {
  await ensureRow(provider, dailyLimit);
  await db.update(providerState)
    .set({ lastErrorAt: new Date(), lastError: message })
    .where(eq(providerState.provider, provider));
}
