"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";

const COLORS = ["#e5e5e5", "#38bdf8", "#f59e0b", "#a78bfa", "#22c55e", "#ef4444", "#ec4899", "#14b8a6", "#eab308", "#8b5cf6"];

type Props = { dates: string[]; series: { ticker: string; normalized: number[] }[] };

export function OverlayChart({ dates, series }: Props) {
  const data = dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    for (const s of series) row[s.ticker] = s.normalized[i];
    return row;
  });
  return (
    <div className="h-96 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s, i) => (
            <Line key={s.ticker} type="monotone" dataKey={s.ticker} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.5} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
