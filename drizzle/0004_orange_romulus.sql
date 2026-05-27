CREATE TABLE "profiles" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "profiles_name_unique" UNIQUE("name")
);
--> statement-breakpoint
INSERT INTO "profiles" ("name") VALUES ('test');
--> statement-breakpoint
ALTER TABLE "watchlist" ADD COLUMN "profile_id" uuid;
--> statement-breakpoint
UPDATE "watchlist" SET "profile_id" = (SELECT "id" FROM "profiles" WHERE "name" = 'test' LIMIT 1) WHERE "profile_id" IS NULL;
--> statement-breakpoint
ALTER TABLE "watchlist" ALTER COLUMN "profile_id" SET NOT NULL;
--> statement-breakpoint
ALTER TABLE "watchlist" ADD CONSTRAINT "watchlist_profile_id_profiles_id_fk" FOREIGN KEY ("profile_id") REFERENCES "public"."profiles"("id") ON DELETE cascade ON UPDATE no action;
--> statement-breakpoint
ALTER TABLE "watchlist" DROP CONSTRAINT "watchlist_ticker_unique";
--> statement-breakpoint
ALTER TABLE "watchlist" ADD CONSTRAINT "uq_watchlist_profile_ticker" UNIQUE("profile_id","ticker");
