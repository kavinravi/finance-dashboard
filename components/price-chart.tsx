"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; ma20: (number | null)[]; ma50: (number | null)[] };

export function PriceChart({ bars, ma20, ma50 }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, close: b.close, ma20: ma20[i], ma50: ma50[i] }));
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Line type="monotone" dataKey="close" stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="ma20" stroke="#38bdf8" dot={false} strokeWidth={1} />
          <Line type="monotone" dataKey="ma50" stroke="#f59e0b" dot={false} strokeWidth={1} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
