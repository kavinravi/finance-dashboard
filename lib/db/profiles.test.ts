import { describe, it, expect, afterAll } from "vitest";
import { createProfile, getProfileById, renameProfile, deleteProfile, listProfiles } from "@/lib/db/profiles";

const created: string[] = [];
afterAll(async () => { for (const id of created) await deleteProfile(id); });

describe("profiles repo (integration, live Neon)", () => {
  it("creates, fetches, renames, and deletes a profile", async () => {
    const p = await createProfile(`unit-${Date.now()}`);
    created.push(p.id);
    expect(await getProfileById(p.id)).toMatchObject({ id: p.id });

    const newName = `unit-renamed-${Date.now()}`;
    await renameProfile(p.id, newName);
    expect((await getProfileById(p.id))?.name).toBe(newName);

    await deleteProfile(p.id);
    created.pop();
    expect(await getProfileById(p.id)).toBeNull();

    expect(Array.isArray(await listProfiles())).toBe(true);
  });
});
