ALTER TABLE "daily_memos" ADD COLUMN "lookback_days" integer;
--> statement-breakpoint
UPDATE "daily_memos" SET "lookback_days" = 7 WHERE "lookback_days" IS NULL;
--> statement-breakpoint
ALTER TABLE "daily_memos" ALTER COLUMN "lookback_days" SET NOT NULL;
--> statement-breakpoint
ALTER TABLE "daily_memos" DROP CONSTRAINT "uq_memo_company_date";
--> statement-breakpoint
ALTER TABLE "daily_memos" ADD CONSTRAINT "uq_memo_company_date_lookback" UNIQUE("company_id","memo_date","lookback_days");
