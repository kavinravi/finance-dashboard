"use client";
import { ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; macdLine: number[]; macdSignal: number[]; macdHistogram: number[] };

export function MacdChart({ bars, macdLine, macdSignal, macdHistogram }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, macd: macdLine[i], signal: macdSignal[i], hist: macdHistogram[i] }));
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="hist" fill="#525252" />
          <Line type="monotone" dataKey="macd" stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="signal" stroke="#38bdf8" dot={false} strokeWidth={1} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
