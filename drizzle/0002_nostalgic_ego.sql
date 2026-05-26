CREATE TABLE "company_fundamentals" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"company_id" uuid NOT NULL,
	"concepts_json" jsonb NOT NULL,
	"fiscal_year" integer,
	"income_period_end" date,
	"balance_sheet_as_of" date,
	"filing_form" text,
	"filed_at" date,
	"source" text NOT NULL,
	"fetched_at" timestamp with time zone DEFAULT now() NOT NULL,
	"expires_at" timestamp with time zone NOT NULL,
	CONSTRAINT "uq_fundamentals_company" UNIQUE("company_id")
);
--> statement-breakpoint
ALTER TABLE "companies" ADD COLUMN "cik" text;--> statement-breakpoint
ALTER TABLE "company_fundamentals" ADD CONSTRAINT "company_fundamentals_company_id_companies_id_fk" FOREIGN KEY ("company_id") REFERENCES "public"."companies"("id") ON DELETE no action ON UPDATE no action;