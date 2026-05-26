import { z } from "zod";

const schema = z.object({
  DATABASE_URL: z.string().url(),
  FMP_API_KEY: z.string().min(1),
  FMP_DAILY_LIMIT: z.coerce.number().int().positive().default(250),
  FINNHUB_API_KEY: z.string().min(1).optional(), // used in SP2
  GEMINI_API_KEY: z.string().min(1).optional(),  // used in SP2
});

const parsed = schema.safeParse({
  DATABASE_URL: process.env.DATABASE_URL,
  FMP_API_KEY: process.env.FMP_API_KEY,
  FMP_DAILY_LIMIT: process.env.FMP_DAILY_LIMIT,
  FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
  GEMINI_API_KEY: process.env.GEMINI_API_KEY,
});

if (!parsed.success) {
  const fields = parsed.error.issues.map((i) => i.path.join(".")).join(", ");
  throw new Error(`Invalid or missing environment variables: ${fields}. Check your .env against .env.example.`);
}

export const env = parsed.data;
