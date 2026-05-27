import { describe, it, expect, afterAll } from "vitest";
import { createProfile, getProfileById, renameProfile, deleteProfile, listProfiles } from "@/lib/db/profiles";

const created: string[] = [];
afterAll(async () => { for (const id of created) await deleteProfile(id); }); // backstop; no-op if already deleted

describe("profiles repo (integration, live Neon)", () => {
  it("creates, fetches, lists, renames, and deletes a profile", async () => {
    const p = await createProfile(`unit-${Date.now()}`);
    created.push(p.id);
    expect(await getProfileById(p.id)).toMatchObject({ id: p.id });
    expect((await listProfiles()).some((row) => row.id === p.id)).toBe(true);

    const newName = `unit-renamed-${Date.now()}`;
    await renameProfile(p.id, newName);
    expect((await getProfileById(p.id))?.name).toBe(newName);

    await deleteProfile(p.id);
    expect(await getProfileById(p.id)).toBeNull();
    expect((await listProfiles()).some((row) => row.id === p.id)).toBe(false);
  });
});
