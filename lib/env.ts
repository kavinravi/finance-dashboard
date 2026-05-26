import { z } from "zod";

const schema = z.object({
  DATABASE_URL: z.string().url(),
  FMP_API_KEY: z.string().min(1),
  FMP_DAILY_LIMIT: z.coerce.number().int().positive().default(250),
  FINNHUB_API_KEY: z.string().min(1).optional(), // used in SP2
  GEMINI_API_KEY: z.string().min(1).optional(),  // used in SP2
  GEMINI_MODEL: z.string().min(1).default("gemini-3.5-flash"),
  GEMINI_PREVIEW_MODEL: z.string().min(1).optional(),
  GEMINI_DAILY_LIMIT: z.coerce.number().int().positive().default(200),
  SEC_USER_AGENT: z.string().min(1).optional(), // SEC requires a contact UA; feature degrades if absent
  APP_PASSWORD: z.string().min(1).optional(),     // SP4 gate password; enforced at runtime by middleware
  SESSION_SECRET: z.string().min(1).optional(),   // SP4 HMAC key for the session cookie
});

const parsed = schema.safeParse({
  DATABASE_URL: process.env.DATABASE_URL,
  FMP_API_KEY: process.env.FMP_API_KEY,
  FMP_DAILY_LIMIT: process.env.FMP_DAILY_LIMIT,
  FINNHUB_API_KEY: process.env.FINNHUB_API_KEY,
  GEMINI_API_KEY: process.env.GEMINI_API_KEY,
  GEMINI_MODEL: process.env.GEMINI_MODEL,
  GEMINI_PREVIEW_MODEL: process.env.GEMINI_PREVIEW_MODEL,
  GEMINI_DAILY_LIMIT: process.env.GEMINI_DAILY_LIMIT,
  SEC_USER_AGENT: process.env.SEC_USER_AGENT,
  APP_PASSWORD: process.env.APP_PASSWORD,
  SESSION_SECRET: process.env.SESSION_SECRET,
});

if (!parsed.success) {
  const fields = parsed.error.issues.map((i) => i.path.join(".")).join(", ");
  throw new Error(`Invalid or missing environment variables: ${fields}. Check your .env against .env.example.`);
}

export const env = parsed.data;
