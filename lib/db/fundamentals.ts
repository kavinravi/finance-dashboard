import { db } from "./client";
import { companyFundamentals } from "./schema";
import { eq } from "drizzle-orm";
import type { FundamentalConcepts, FundamentalsMeta } from "@/lib/types";

export type FundamentalsRow = typeof companyFundamentals.$inferSelect;

export async function getFundamentalsSnapshot(companyId: string): Promise<FundamentalsRow | undefined> {
  const [row] = await db.select().from(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
  return row;
}

export async function upsertFundamentalsSnapshot(input: {
  companyId: string;
  conceptsJson: FundamentalConcepts;
  meta: FundamentalsMeta;
  source: string;
  expiresAt: Date;
}): Promise<void> {
  const values = {
    companyId: input.companyId,
    conceptsJson: input.conceptsJson,
    fiscalYear: input.meta.fiscalYear,
    incomePeriodEnd: input.meta.incomePeriodEnd,
    balanceSheetAsOf: input.meta.balanceSheetAsOf,
    filingForm: input.meta.filingForm,
    filedAt: input.meta.filedAt,
    source: input.source,
    fetchedAt: new Date(),
    expiresAt: input.expiresAt,
  };
  await db.insert(companyFundamentals).values(values).onConflictDoUpdate({
    target: companyFundamentals.companyId,
    set: { ...values, fetchedAt: new Date() },
  });
}
