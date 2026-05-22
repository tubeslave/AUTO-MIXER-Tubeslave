# ADR: Automixer main Dashboard frontend

## 1. Context

AUTO-MIXER needs a production-ready main Dashboard surface for operators. The current
frontend already has a Vite/React/TypeScript prototype, Product Layer adapters, mock
fallback data, and hash routing. The requested work is frontend-only and must not
connect real OSC write commands or change backend behavior.

## 2. Options considered

- Build the Dashboard inside the existing TypeScript app shell.
- Replace the shell with a new single-purpose Dashboard app.
- Extend the older `App.js` control surface.

## 3. Decision

Build the Dashboard as a dedicated TypeScript component slice inside the existing
Vite/Tailwind app shell and keep the route at `#/dashboard`.

## 4. Why this won

This preserves the active frontend entrypoint, current Product Layer adapters, mode
guard copy, mock fallback behavior, and existing navigation. It also keeps the change
frontend-only and avoids introducing new dependencies or live mixer write paths.

## 5. Rejected alternatives

- A full shell replacement would increase the blast radius and risk losing existing
  backend adapter work.
- Editing the older `App.js` would target a non-active entrypoint for the Vite app.
- Pulling in charting libraries is unnecessary while telemetry is mocked and simple
  SVG/div visualizations are sufficient.

## 6. Implementation plan

- Add Dashboard mock data in a separate frontend file.
- Add reusable Dashboard and Director panel components.
- Wire the Dashboard route to the new components.
- Enhance the top status bar with session/status/mode/safe/mixer/user signals.
- Keep all accept/reject/transport interactions local mock actions.

## 7. Test plan

- Run `npm --prefix frontend run build`.
- Start the Vite dev server and inspect `#/dashboard`.
- Verify desktop-first layout and responsive containment.

## 8. Risks and rollback

Risk is mostly visual layout regression in the active frontend shell. Rollback is to
restore the previous `DashboardPanels.tsx` route wiring and remove the Dashboard-specific
mock/component files. No backend state or mixer state is changed.
