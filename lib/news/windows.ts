// Selectable news-lookback windows (days back) for the sentiment memo.
export const NEWS_WINDOWS = [1, 5] as const;
export type NewsWindow = (typeof NEWS_WINDOWS)[number];
export const DEFAULT_WINDOW: NewsWindow = 5;

// Coerce arbitrary input (query param, etc.) to a valid window; falls back to the default.
export function parseWindow(raw: string | number | null | undefined): NewsWindow {
  const n = typeof raw === "string" ? Number(raw) : raw;
  return (NEWS_WINDOWS as readonly number[]).includes(n as number) ? (n as NewsWindow) : DEFAULT_WINDOW;
}
