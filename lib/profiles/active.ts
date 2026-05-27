import { cookies } from "next/headers";
import { PROFILE_COOKIE } from "@/lib/auth/profile-gate";
import { getProfileById } from "@/lib/db/profiles";

// Server-only: the active profile id from the session cookie, or null if unset OR the
// cookie points at a profile that no longer exists (stale after a delete on another device).
export async function getActiveProfileId(): Promise<string | null> {
  const jar = await cookies();
  const id = jar.get(PROFILE_COOKIE)?.value;
  if (!id) return null;
  const profile = await getProfileById(id);
  return profile ? id : null;
}
