import { getAllProviderStates } from "@/lib/db/provider-state";
import { env } from "@/lib/env";
import { PruneButton } from "@/components/prune-button";

export const dynamic = "force-dynamic";

function fmt(d: Date | null): string {
  return d ? new Date(d).toISOString().replace("T", " ").slice(0, 19) + "Z" : "—";
}

export default async function HealthPage() {
  let rows: Awaited<ReturnType<typeof getAllProviderStates>> = [];
  let dbOk = true;
  try {
    rows = await getAllProviderStates();
  } catch {
    dbOk = false;
  }

  const configured: Record<string, boolean> = {
    FINNHUB_API_KEY: Boolean(env.FINNHUB_API_KEY),
    GEMINI_API_KEY: Boolean(env.GEMINI_API_KEY),
    SEC_USER_AGENT: Boolean(env.SEC_USER_AGENT),
    APP_PASSWORD: Boolean(env.APP_PASSWORD),
  };

  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <h1 className="text-2xl font-semibold">System health</h1>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Configuration</h2>
        <p className="text-sm">Database: {dbOk ? "reachable" : "unreachable"}</p>
        <p className="text-sm">Server time: {fmt(new Date())}</p>
        <ul className="text-sm text-neutral-300">
          {Object.entries(configured).map(([k, v]) => (
            <li key={k}>{k}: {v ? "set" : "missing"}</li>
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Providers</h2>
        {dbOk ? (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-neutral-800 text-left">
                <th className="py-1 pr-4">Provider</th>
                <th className="py-1 pr-4">Calls today</th>
                <th className="py-1 pr-4">Limit</th>
                <th className="py-1 pr-4">Last success</th>
                <th className="py-1 pr-4">Last error</th>
                <th className="py-1 pr-4">Error msg</th>
                <th className="py-1 pr-4">Resets</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.provider} className="border-b border-neutral-900">
                  <td className="py-1 pr-4">{r.provider}</td>
                  <td className="py-1 pr-4">{r.callsToday}</td>
                  <td className="py-1 pr-4">{r.dailyLimit}</td>
                  <td className="py-1 pr-4">{fmt(r.lastSuccessAt)}</td>
                  <td className="py-1 pr-4">{fmt(r.lastErrorAt)}</td>
                  <td className="max-w-[16rem] truncate py-1 pr-4">{r.lastError ?? "—"}</td>
                  <td className="py-1 pr-4">{fmt(r.resetAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-red-400">Database unreachable — cannot read provider state.</p>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-medium">Maintenance</h2>
        <PruneButton />
      </section>
    </main>
  );
}
