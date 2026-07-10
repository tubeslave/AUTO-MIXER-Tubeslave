# Codex proposal: frontend live cockpit

## 1. Short thesis

Turn the existing cockpit into a truthful readback-first console surface. Live
data must replace presentation fallbacks, while unsupported writes must become
visibly unavailable instead of simulated.

## 2. What I understood

The strongest existing asset is the dense sound-engineer layout. Its primary
defect is semantic: live and demo state are mixed, several controls are local
simulations, and late WebSocket snapshots can overwrite newer runtime status.

## 3. Proposed solution

- Autoconnect to the Product Layer unless demo mode is explicitly requested.
- Subscribe to connection, mixer, capture and meter events.
- Add availability/provenance fields to the adapters and render `NO SIGNAL` or
  `Unavailable` rather than synthetic numbers.
- Replace the fake transport with a runtime control bar bound to backend
  Refresh, Analyze and audio-meter actions.
- Make WING connection observation-only by default.
- Disable unavailable automation modes and remove inert system controls.
- Give mobile users the same authority, console and audio truth in a compact
  layout.
- Add a regression contract for every removed truth gap.

## 4. Likely files to touch

`frontend/src/services/`, dashboard and prototype components, system/director
views, frontend tests/E2E, the Goal, ADR and frontend audit report.

## 5. Alternatives considered

- Keep the mock dashboard and add a small live badge: rejected because fake
  metrics remain actionable misinformation.
- Make every fader interactive immediately: rejected because the backend does
  not grant broad live-write authority.
- Redesign the visual language: rejected because the existing console-density
  design is appropriate and should be refined rather than replaced.

## 6. Risks

- High-rate meter traffic can flood logs or rerender unnecessarily.
- Stale snapshots can regress newer event state.
- A control that looks enabled can imply authority the backend does not grant.
- The single production bundle is slightly above the Vite warning threshold.

## 7. Test plan

Frontend unit/regression tests, production build, nine-route E2E suite,
targeted backend audio/meter tests, real WING browser QA on desktop/mobile,
Start/Stop race testing, clean browser console and screenshot comparison.

## 8. Where the other agent may disagree

An alternate reviewer may prefer editable faders. The safer decision is to keep
them as accurate readback until supervised write/readback/rollback evidence is
available for the exact control family.
