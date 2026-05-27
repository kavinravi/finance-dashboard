import { cookies } from "next/headers";
import { PROFILE_COOKIE } from "@/lib/auth/profile-gate";

// Server-only: the active profile id from the session cookie (null if unset).
export async function getActiveProfileId(): Promise<string | null> {
  const jar = await cookies();
  return jar.get(PROFILE_COOKIE)?.value ?? null;
}
