"use client";
import { useEffect } from "react";
import Link from "next/link";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="mx-auto max-w-4xl px-4 pt-24 text-center">
      <h1 className="text-xl font-semibold">Something went wrong</h1>
      <p className="mt-2 text-neutral-400">
        We couldn’t load this page — a data source or the database may be temporarily unavailable.
      </p>
      <div className="mt-6 flex justify-center gap-3">
        <button onClick={reset} className="rounded bg-neutral-200 px-4 py-2 text-sm font-medium text-neutral-900">
          Try again
        </button>
        <Link href="/" className="rounded px-4 py-2 text-sm text-neutral-400 ring-1 ring-neutral-800">
          Back to search
        </Link>
      </div>
    </main>
  );
}
