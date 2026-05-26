"use client";
import { useState } from "react";
import { PriceChart } from "./price-chart";
import { RsiChart } from "./rsi-chart";
import { MacdChart } from "./macd-chart";
import { sliceByRange, downsample, type ChartRange, type SliceableIndicators } from "@/lib/charts/range";
import type { PriceBar } from "@/lib/types";

const MAX_POINTS = 800; // Recharts can't draw a line for ~thousands of points; cap for long ranges.

const PRESETS = [
  { key: "1m", label: "1M" }, { key: "3m", label: "3M" }, { key: "6m", label: "6M" },
  { key: "ytd", label: "YTD" }, { key: "1y", label: "1Y" }, { key: "all", label: "ALL" },
] as const;

export function TickerCharts({ bars, indicators }: { bars: PriceBar[]; indicators: SliceableIndicators }) {
  const [range, setRange] = useState<ChartRange>("1y");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const today = new Date().toISOString().slice(0, 10);
  const windowed = sliceByRange(bars, indicators, range, today);
  const sliced = downsample(windowed.bars, windowed.indicators, MAX_POINTS);

  const presetActive = (k: string) => typeof range !== "object" && range === k;
  const btn = (active: boolean) =>
    `rounded px-2 py-1 text-xs ${active ? "bg-neutral-200 text-neutral-900" : "bg-neutral-900 text-neutral-300 ring-1 ring-neutral-800 hover:bg-neutral-800"}`;

  function applyCustom(nextFrom: string, nextTo: string) {
    setFrom(nextFrom);
    setTo(nextTo);
    if (nextFrom && nextTo) setRange({ from: nextFrom, to: nextTo });
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => (
          <button key={p.key} onClick={() => setRange(p.key)} className={btn(presetActive(p.key))}>{p.label}</button>
        ))}
        <span className="ml-2 flex items-center gap-1 text-xs text-neutral-500">
          <input type="date" value={from} max={to || today} onChange={(e) => applyCustom(e.target.value, to)}
            className="rounded bg-neutral-900 px-2 py-1 text-neutral-200 ring-1 ring-neutral-800 [color-scheme:dark]" />
          <span>→</span>
          <input type="date" value={to} min={from} max={today} onChange={(e) => applyCustom(from, e.target.value)}
            className="rounded bg-neutral-900 px-2 py-1 text-neutral-200 ring-1 ring-neutral-800 [color-scheme:dark]" />
        </span>
      </div>

      <div><PriceChart bars={sliced.bars} ma20={sliced.indicators.ma20} ma50={sliced.indicators.ma50} /></div>

      <h2 className="mt-6 text-sm font-medium text-neutral-400">RSI (14)</h2>
      <div className="mt-2"><RsiChart bars={sliced.bars} rsi14={sliced.indicators.rsi14} /></div>

      <h2 className="mt-6 text-sm font-medium text-neutral-400">MACD (12/26/9)</h2>
      <div className="mt-2">
        <MacdChart bars={sliced.bars} macdLine={sliced.indicators.macdLine} macdSignal={sliced.indicators.macdSignal} macdHistogram={sliced.indicators.macdHistogram} />
      </div>
    </div>
  );
}
