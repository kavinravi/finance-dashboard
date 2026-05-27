import { db } from "./client";
import { profiles } from "./schema";
import { asc, eq } from "drizzle-orm";

export type ProfileRow = typeof profiles.$inferSelect;

export async function listProfiles(): Promise<ProfileRow[]> {
  return db.select().from(profiles).orderBy(asc(profiles.createdAt));
}

export async function getProfileById(id: string): Promise<ProfileRow | null> {
  const rows = await db.select().from(profiles).where(eq(profiles.id, id)).limit(1);
  return rows[0] ?? null;
}

export async function createProfile(name: string): Promise<ProfileRow> {
  const [row] = await db.insert(profiles).values({ name }).returning();
  return row;
}

export async function renameProfile(id: string, name: string): Promise<void> {
  await db.update(profiles).set({ name }).where(eq(profiles.id, id));
}

export async function deleteProfile(id: string): Promise<void> {
  await db.delete(profiles).where(eq(profiles.id, id));
}
