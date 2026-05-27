"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { PriceBar } from "@/lib/types";

export const MA_COLORS = { ma20: "#38bdf8", ma50: "#f59e0b", ma200: "#a78bfa" } as const;
export type MaVisibility = { ma20: boolean; ma50: boolean; ma200: boolean };

type Props = {
  bars: PriceBar[];
  ma20: (number | null)[];
  ma50: (number | null)[];
  ma200: (number | null)[];
  visible?: MaVisibility;
};

export function PriceChart({ bars, ma20, ma50, ma200, visible }: Props) {
  const v = visible ?? { ma20: true, ma50: true, ma200: true };
  const data = bars.map((b, i) => ({ date: b.date, close: b.close, ma20: ma20[i], ma50: ma50[i], ma200: ma200[i] }));
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} syncId="ticker" margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Line type="monotone" dataKey="close" stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          {v.ma20 && <Line type="monotone" dataKey="ma20" stroke={MA_COLORS.ma20} dot={false} strokeWidth={1} />}
          {v.ma50 && <Line type="monotone" dataKey="ma50" stroke={MA_COLORS.ma50} dot={false} strokeWidth={1} />}
          {v.ma200 && <Line type="monotone" dataKey="ma200" stroke={MA_COLORS.ma200} dot={false} strokeWidth={1} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
