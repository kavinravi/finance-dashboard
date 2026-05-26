"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; rsi14: (number | null)[] };

export function RsiChart({ bars, rsi14 }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, rsi: rsi14[i] }));
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis domain={[0, 100]} ticks={[0, 30, 70, 100]} tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
          <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="rsi" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
