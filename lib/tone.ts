export type ToneLabel = "bearish" | "somewhat_bearish" | "neutral" | "somewhat_bullish" | "bullish";

export function toneLabelText(label: ToneLabel): string {
  const map: Record<ToneLabel, string> = {
    bearish: "Bearish",
    somewhat_bearish: "Somewhat bearish",
    neutral: "Neutral",
    somewhat_bullish: "Somewhat bullish",
    bullish: "Bullish",
  };
  return map[label];
}

export function toneColor(score: number): string {
  if (score <= 30) return "#ef4444"; // red
  if (score <= 45) return "#f59e0b"; // amber
  if (score <= 55) return "#a3a3a3"; // neutral grey
  if (score <= 70) return "#84cc16"; // lime
  return "#22c55e";                  // green
}
