import { SearchBar } from "@/components/search-bar";
import { RecentSearches } from "@/components/recent-searches";

export const dynamic = "force-dynamic";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center px-4 pt-24">
      <h1 className="mb-8 text-2xl font-semibold">Finance Dashboard</h1>
      <SearchBar />
      <RecentSearches />
    </main>
  );
}
