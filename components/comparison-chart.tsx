"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";

type Props = { dates: string[]; primaryTicker: string; comparisonTicker: string; primary: number[]; comparison: number[] };

export function ComparisonChart({ dates, primaryTicker, comparisonTicker, primary, comparison }: Props) {
  const data = dates.map((date, i) => ({ date, [primaryTicker]: primary[i], [comparisonTicker]: comparison[i] }));
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey={primaryTicker} stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey={comparisonTicker} stroke="#38bdf8" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
