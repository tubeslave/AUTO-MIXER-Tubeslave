# Automixer Dashboard Main Proposal

## 1. Short thesis

Implement the requested Dashboard as a TypeScript frontend slice inside the existing
Vite/Tailwind prototype, preserving the current Product Layer adapters while keeping
all controls mock/local until backend telemetry is explicitly connected.

## 2. What I understood

The user wants the primary Automixer operator screen, not a simplified demo. The page
should look and behave like a dense AI live mixing console: top session state, left
navigation, analytics cards, central mixer with channel strips, lower analysis panels,
AI Director right panel, and footer transport/safety controls. Backend and OSC writes
are out of scope.

## 3. Proposed solution

- Add a Dashboard-specific mock data module for analytics, tabs, channel telemetry,
  analyzer charts, director events, suggestions, transport, and integration TODOs.
- Add reusable Dashboard components for cards, meters, channel strips, chart panels,
  the central mixer, and the right AI Director panel.
- Keep the existing hash route `#/dashboard` and wire the new Dashboard workspace into
  `AutomixerViews.tsx`.
- Improve the global top status bar so it shows session, status, mode, safe policy,
  connected mixer, and user.
- Keep actions local through the existing `UiActions.pushToast`, snapshot, and rollback
  helpers.

## 4. Likely files to touch

- `frontend/src/mock/dashboardMock.ts`
- `frontend/src/components/dashboard/AutomixerDashboard.tsx`
- `frontend/src/components/dashboard/DirectorPanel.tsx`
- `frontend/src/components/prototype/AutomixerComponents.tsx`
- `frontend/src/components/prototype/DashboardPanels.tsx`
- `frontend/src/views/AutomixerViews.tsx`
- `Docs/adr/automixer-dashboard-main.md`

## 5. Alternatives considered

- Replace the whole app shell: rejected because the current app already has routing,
  Product Layer adapters, mock fallback, and safety-aware actions.
- Extend the older `App.js`: rejected because Vite now renders `main.tsx` and the
  TypeScript prototype is the active frontend entry.
- Add a charting dependency: rejected because simple SVG/div visualizations satisfy the
  current mock telemetry requirement without dependency risk.

## 6. Risks

- Existing worktree is dirty; edits must avoid reverting unrelated Product Layer and
  backend changes.
- Dense dashboards can overflow on smaller widths; desktop-first responsive constraints
  need explicit minimums and compact behavior.
- UI controls could look like real apply paths; copy and handlers must keep them local
  unless the backend supervised gate is explicitly used later.

## 7. Test plan

- `npm --prefix frontend run build`
- Run the Vite dev server.
- Inspect the Dashboard in browser at desktop and narrower viewport sizes.
- Capture screenshot evidence for the resulting UI.

## 8. Where the other agent may disagree

Kimi may prefer a bigger shell redesign or deeper split into smaller files. I would keep
the first pass scoped to Dashboard-specific components so the existing routing, Product
Layer adapters, and safety behavior remain intact.
