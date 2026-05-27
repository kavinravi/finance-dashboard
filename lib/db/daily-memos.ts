import { db } from "./client";
import { dailyMemos } from "./schema";
import { and, eq } from "drizzle-orm";

export type DailyMemoRow = typeof dailyMemos.$inferSelect;

export async function getMemoForDate(companyId: string, memoDate: string, lookbackDays: number): Promise<DailyMemoRow | undefined> {
  const [row] = await db.select().from(dailyMemos)
    .where(and(
      eq(dailyMemos.companyId, companyId),
      eq(dailyMemos.memoDate, memoDate),
      eq(dailyMemos.lookbackDays, lookbackDays),
    ));
  return row;
}

export async function upsertMemo(row: {
  companyId: string; memoDate: string; lookbackDays: number; model: string; summaryJson: unknown;
  toneLabel: string; toneScore: number; sourceArticleIds: string[]; basedOnArticleCount: number;
}): Promise<void> {
  await db.insert(dailyMemos)
    .values({ ...row, generatedAt: new Date() })
    .onConflictDoUpdate({
      target: [dailyMemos.companyId, dailyMemos.memoDate, dailyMemos.lookbackDays],
      set: {
        model: row.model, summaryJson: row.summaryJson, toneLabel: row.toneLabel,
        toneScore: row.toneScore, sourceArticleIds: row.sourceArticleIds,
        basedOnArticleCount: row.basedOnArticleCount, generatedAt: new Date(),
      },
    });
}
