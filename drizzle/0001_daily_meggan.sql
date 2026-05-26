CREATE TABLE "articles" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"company_id" uuid NOT NULL,
	"source" text NOT NULL,
	"source_article_id" text,
	"url" text NOT NULL,
	"url_hash" text NOT NULL,
	"title" text NOT NULL,
	"summary" text,
	"published_at" timestamp with time zone NOT NULL,
	"image_url" text,
	"related" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"expires_at" timestamp with time zone,
	CONSTRAINT "uq_article_company_urlhash" UNIQUE("company_id","url_hash")
);
--> statement-breakpoint
CREATE TABLE "daily_memos" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"company_id" uuid NOT NULL,
	"memo_date" date NOT NULL,
	"model" text NOT NULL,
	"summary_json" jsonb NOT NULL,
	"tone_label" text NOT NULL,
	"tone_score" integer NOT NULL,
	"source_article_ids" jsonb NOT NULL,
	"based_on_article_count" integer NOT NULL,
	"generated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "uq_memo_company_date" UNIQUE("company_id","memo_date")
);
--> statement-breakpoint
ALTER TABLE "articles" ADD CONSTRAINT "articles_company_id_companies_id_fk" FOREIGN KEY ("company_id") REFERENCES "public"."companies"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "daily_memos" ADD CONSTRAINT "daily_memos_company_id_companies_id_fk" FOREIGN KEY ("company_id") REFERENCES "public"."companies"("id") ON DELETE no action ON UPDATE no action;