import { db } from "./client";
import { priceBarsDaily } from "./schema";
import { eq, asc } from "drizzle-orm";
import type { PriceBar } from "@/lib/types";

export async function upsertBars(companyId: string, bars: PriceBar[], source: string): Promise<void> {
  if (bars.length === 0) return;
  const rows = bars.map((b) => ({
    companyId, date: b.date, open: b.open, high: b.high, low: b.low, close: b.close,
    adjClose: b.adjClose, volume: b.volume, source,
  }));
  for (let i = 0; i < rows.length; i += 500) {
    await db.insert(priceBarsDaily).values(rows.slice(i, i + 500)).onConflictDoNothing();
  }
}

export async function getBars(companyId: string): Promise<PriceBar[]> {
  const rows = await db.select().from(priceBarsDaily)
    .where(eq(priceBarsDaily.companyId, companyId)).orderBy(asc(priceBarsDaily.date));
  return rows.map((r) => ({
    date: r.date, open: r.open, high: r.high, low: r.low, close: r.close,
    adjClose: r.adjClose, volume: r.volume,
  }));
}
