import { toneColor, toneLabelText, type ToneLabel } from "@/lib/tone";

export function ToneMeter({ label, score }: { label: ToneLabel; score: number }) {
  const clamped = Math.max(0, Math.min(100, score));
  return (
    <div className="my-3">
      <div className="flex items-center justify-between text-xs text-neutral-400">
        <span>News Tone</span>
        <span>{score}/100 · {toneLabelText(label)}</span>
      </div>
      <div className="mt-1 h-2 w-full rounded bg-neutral-800">
        <div className="h-2 rounded" style={{ width: `${clamped}%`, background: toneColor(score) }} />
      </div>
    </div>
  );
}
