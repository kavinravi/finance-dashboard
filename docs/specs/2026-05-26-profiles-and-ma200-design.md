# Profile Layer + 200-Day MA Toggles — Design Spec

**Date:** 2026-05-26
**Branch:** `rebuild-nextjs-mvp`
**Status:** Approved design; ready for implementation planning.

Two user-requested changes to the deployed dashboard:

1. **200-day moving average + line toggles** — add a third MA line and let the user show/hide each MA line via checkboxes in the chart controls.
2. **Profile layer** — a selectable identity (Netflix-style "who's watching?" picker) whose only job is to scope the **watchlist**, so testing on one profile never disturbs another profile's list. All other cached data stays shared.

It also captures one **operational fix** (not code) for the user-approval friction during sharing — see §6.

---

## 1. Feature 1 — 200-day MA + line toggles

### 1.1 What exists today
- `lib/services/price-service.ts:94` computes `ma10/ma20/ma50` via `sma(closes, period)`; history is fetched **max-available** (see `price-fetch-window.ts`), so a 200-bar window is available for established tickers.
- `lib/charts/range.ts` defines `SliceableIndicators` and slices/downsamples each indicator field **explicitly by name** (`sliceByRange` ~line 48, `downsample` ~line 71).
- `app/ticker/[symbol]/page.tsx` passes a `SliceableIndicators` subset (currently `ma20`, `ma50`) into `TickerCharts`.
- `components/ticker-charts.tsx` renders the controls row (presets + date pickers, lines 36–46) and `<PriceChart ma20 ma50 />` (line 49).
- `components/price-chart.tsx` (Recharts) draws price + two MA `<Line>`s: 20 = blue `#38bdf8`, 50 = orange `#f59e0b`.

### 1.2 Changes
- **Compute:** add `ma200: sma(closes, 200)` to the indicators object in `price-service.ts` (and to `TickerData.indicators`).
- **Thread through:** add `ma200: (number | null)[]` to `SliceableIndicators`, and add it to the explicit field lists in both `sliceByRange` and `downsample`. Add `ma200` to the indicators object built in `page.tsx`.
- **PriceChart:** accept `ma200` plus a `visible` flag set `{ ma20, ma50, ma200 }`; render each `<Line>` only when its flag is true. New color: 200 = violet `#a78bfa`.
- **Controls (`ticker-charts.tsx`):** add three checkboxes (20 / 50 / 200) to the existing controls row, after the date pickers (fills the trailing whitespace). State defaults **all three on**, persisted to `localStorage` under `fd_ma_visible` (per device). Each checkbox label includes a small color swatch matching its line, so the row doubles as a legend.

### 1.3 Behavior notes
- The 200d line is valid across **1Y / ALL** for established tickers. Where there are fewer than 200 prior bars (young ticker, or a short range like 1M), the SMA values are `null` and Recharts simply draws no segment there — expected, not an error.
- Toggling is pure client state; no refetch.

---

## 2. Feature 2 — Profile layer (watchlist scoping)

**Reframe:** the password login already exists (SP4: `lib/auth/`, `proxy.ts`, `/login`) and is **unchanged**. This feature adds *profiles* and makes the watchlist per-profile. Memos, price bars, articles, and fundamentals are already global (keyed by ticker) and **stay shared** — no change.

### 2.1 Data model
- **New `profiles` table:** `id` (uuid, pk), `name` (text, **unique**), `createdAt`.
- **`watchlist` table:** add `profileId` (uuid, FK → `profiles.id`, `on delete cascade`). Replace the `ticker`-unique constraint with a composite **unique `(profileId, ticker)`** (same ticker may appear in different profiles).
- **Migration** (`drizzle/`): create `profiles`; **seed exactly one profile named `test`**; add `profileId` to `watchlist`; **backfill all existing watchlist rows** with the seeded profile's id (nothing is lost); add the new composite unique; drop the old `ticker`-unique.

### 2.2 Active-profile state
- An **`fd_profile` session cookie** (no `Max-Age`/`Expires`) holds the selected profile id. Because a session cookie clears when the browser is fully closed, **each fresh open re-prompts** ("always prompt on load"), while a refresh or in-session navigation does **not** nag. Different devices stay independent (dad's device vs. the test device).
- The cookie holds only a profile id, lives behind the password gate, and is validated server-side against existing profiles — so no HMAC signing is needed (unlike the auth session token). `httpOnly`, `sameSite=lax`.

### 2.3 Selection gate (mirrors the existing login gate)
In `proxy.ts`, after auth passes: if the request is a page navigation (not `/api/*`, not assets), there is no valid `fd_profile` cookie, and the path is not already `/select-profile`, redirect to `/select-profile`. Exemptions (auth-gated but profile-exempt): `/select-profile`, `/api/profiles*`, `/api/profile/select`. The watchlist API additionally enforces a valid profile (see §2.6).

### 2.4 Picker page `/select-profile` (Netflix "who's watching?" aesthetic)
- Centered heading, a row of **profile tiles** (colored square + name; **no images**), and a large **"Add Profile"** tile on the right.
- Selecting a tile → `POST /api/profile/select` → sets `fd_profile` → returns to the page the user came from (safe-path, reusing `safeNextPath`), default home.
- **"Add Profile"** reveals an inline name field → `POST /api/profiles` → selects the new profile.
- A **"Manage profiles"** toggle flips tiles into edit mode: each tile shows a **one-click delete** control (and rename). Delete calls `DELETE /api/profiles/[id]`.
  - Because deleting a profile **cascades its watchlist**, delete shows a single lightweight confirm ("Delete 'X' and its watchlist?"). *(If the user wants truly no-confirm deletion, drop the confirm — flagged as a choice.)*
- Tile colors are assigned from a small fixed palette by index (deterministic), so tiles look distinct without storing a color.

### 2.5 In-app switcher (`components/site-header.tsx`)
- The active profile name is read **server-side** from the `fd_profile` cookie (in `SiteHeader` or a thin server wrapper) and rendered as a chip.
- An interactive client subcomponent provides a dropdown to **switch** (calls `/api/profile/select`, then refreshes) and a **"Manage profiles"** link to `/select-profile`.

### 2.6 APIs and data layer
- `GET /api/profiles` — list profiles. `POST /api/profiles` — create `{ name }` (trim, reject empty/duplicate).
- `PATCH /api/profiles/[id]` — rename. `DELETE /api/profiles/[id]` — delete (watchlist cascades).
- `POST /api/profile/select` — body `{ id }`; validate the id exists, set the `fd_profile` cookie.
- `/api/watchlist` (GET/POST/DELETE) reads `fd_profile` from the cookie and **scopes every query by `profileId`**; if the cookie is missing or stale/invalid, respond so the client routes to `/select-profile`.
- `lib/db/watchlist.ts` — `getWatchlist`, `addToWatchlist`, `removeFromWatchlist` all take `profileId`.
- `lib/services/watchlist-service.ts` — `getWatchlistOverlay(profileId)`.

### 2.7 Edge cases
- **Zero profiles** (user deleted them all): picker shows only "Add Profile"; the watchlist is unavailable until one exists. The migration guarantees at least the seeded `test` profile initially.
- **Stale `fd_profile`** (cookie points at a deleted profile): treated as "no profile" → redirect to picker.

---

## 3. Out of scope (YAGNI)
- **Per-profile passwords** — the shared password stays the only credential, as requested.
- **Scoping recent searches or memos** — they remain global/shared; only the watchlist becomes per-profile.
- **Profile avatars/images** — tiles use colors + names only.

---

## 4. Testing
- **Unit:** `sma(closes, 200)` correctness (and `null` fill before bar 200); `sliceByRange`/`downsample` include `ma200`.
- **Component/E2E (Playwright):**
  - Toggle each MA checkbox → corresponding line shows/hides; reload preserves toggles (localStorage).
  - Create a profile; add a ticker under profile A; switch to profile B → its watchlist is empty/independent; switch back → A's ticker is present.
  - Delete a profile → it disappears and its watchlist rows are gone (cascade); other profiles unaffected.
  - New browser session (cleared session cookie) → redirected to `/select-profile`.
- **Live smoke (required before "done", per project rule):** with the dev server running, exercise a real watchlist add/remove under a profile against the live DB — mocked tests have masked broken live paths before.

---

## 5. File-change summary
**Feature 1:** `lib/services/price-service.ts`, `lib/charts/range.ts`, `app/ticker/[symbol]/page.tsx`, `components/ticker-charts.tsx`, `components/price-chart.tsx`.
**Feature 2:** `lib/db/schema.ts` (+ new `drizzle/` migration), `lib/db/watchlist.ts`, `lib/services/watchlist-service.ts`, `app/api/watchlist/route.ts`, new `app/api/profiles/route.ts` + `app/api/profiles/[id]/route.ts` + `app/api/profile/select/route.ts`, new `app/select-profile/page.tsx`, `components/site-header.tsx` (+ client switcher subcomponent), `proxy.ts`.

---

## 6. Operational note — user-approval friction (not code)
The "enter a username, wait for the owner to approve" delay during sharing is **Vercel Deployment Protection ("Vercel Authentication")** sitting in front of the app, not anything in this repo (the app's own shared-password gate was built specifically to avoid it — see `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md:413`). Fix once in the dashboard: **Vercel → project → Settings → Deployment Protection → set "Vercel Authentication" to Disabled (at least for Production) → Save.** The app's password gate continues to protect the deployment, so anyone with the password gets in with no per-user approval.
