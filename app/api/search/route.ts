import { type NextRequest } from "next/server";
import { z } from "zod";
import { resolveQuery } from "@/lib/services/search-service";

const Query = z.object({ q: z.string().min(1).max(64) });

export async function GET(request: NextRequest) {
  const parsed = Query.safeParse({ q: request.nextUrl.searchParams.get("q") ?? "" });
  if (!parsed.success) return Response.json({ error: "Invalid query" }, { status: 400 });
  try {
    return Response.json({ results: await resolveQuery(parsed.data.q) });
  } catch (e) {
    return Response.json({ error: "Search failed", detail: String(e) }, { status: 502 });
  }
}
