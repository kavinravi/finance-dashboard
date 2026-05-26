import { type NextRequest } from "next/server";
import { z } from "zod";
import { compareTickers } from "@/lib/services/comparison-service";

const Schema = z.object({
  primary: z.string().regex(/^[A-Za-z.\-]{1,10}$/),
  comparison: z.string().regex(/^[A-Za-z.\-]{1,10}$/),
  range: z.enum(["1m", "3m", "6m", "ytd", "1y"]).default("1y"),
});

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const parsed = Schema.safeParse({
    primary: sp.get("primary"), comparison: sp.get("comparison"), range: sp.get("range") ?? undefined,
  });
  if (!parsed.success) return Response.json({ error: "Invalid params" }, { status: 400 });
  try {
    const { primary, comparison, range } = parsed.data;
    return Response.json(await compareTickers(primary.toUpperCase(), comparison.toUpperCase(), range));
  } catch (e) {
    return Response.json({ error: "Compare failed", detail: String(e) }, { status: 502 });
  }
}
