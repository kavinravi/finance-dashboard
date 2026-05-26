import { formatPercent } from "@/lib/formatters";
import type { PeriodReturns } from "@/lib/types";

const LABELS: [keyof PeriodReturns, string][] = [
  ["oneDay", "1D"], ["fiveDay", "5D"], ["oneMonth", "1M"],
  ["threeMonth", "3M"], ["sixMonth", "6M"], ["ytd", "YTD"], ["oneYear", "1Y"],
];

export function ReturnsTable({ returns }: { returns: PeriodReturns }) {
  return (
    <div className="grid grid-cols-7 gap-px overflow-hidden rounded ring-1 ring-neutral-800">
      {LABELS.map(([key, label]) => {
        const v = returns[key];
        const color = v === null ? "text-neutral-500" : v >= 0 ? "text-emerald-400" : "text-red-400";
        return (
          <div key={key} className="bg-neutral-900 px-2 py-2 text-center">
            <div className="text-[10px] uppercase text-neutral-500">{label}</div>
            <div className={`text-sm font-medium ${color}`}>{formatPercent(v)}</div>
          </div>
        );
      })}
    </div>
  );
}
