import { listProfiles } from "@/lib/db/profiles";
import { ProfilePicker } from "@/components/profile-picker";

export const dynamic = "force-dynamic";
export const metadata = { title: "Who's looking? · Finance Dashboard" };

export default async function SelectProfilePage({ searchParams }: { searchParams: Promise<{ next?: string }> }) {
  const { next } = await searchParams;
  const all = await listProfiles();
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6">
      <h1 className="mb-12 text-4xl font-semibold text-neutral-100">Who&apos;s looking?</h1>
      <ProfilePicker profiles={all.map((p) => ({ id: p.id, name: p.name }))} next={next ?? "/"} />
    </main>
  );
}
