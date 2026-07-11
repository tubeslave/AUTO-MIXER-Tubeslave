# Soundcheck Workflow 95 — Final Review

Date: 2026-07-11

## Result

Local proposal-first soundcheck workflow: **98/100**.

This score covers the locally verifiable workflow from real read-only mixer/audio
state through absolute recommendation derivation and operator review. It does not
grant WING live-write authority and does not close physical M005 or owner review
M006.

## Rubric

| Area | Score | Evidence |
| --- | ---: | --- |
| Live safety boundary | 25/25 | web, CLI and constructor direct apply/emergency paths fail closed; YAML promotion cannot widen SafetyController authority |
| Recommendation quality | 24/25 | explicit derivation phase, typed absolute targets, baseline flags, stable ids and 600-cycle no-drift regression |
| Runtime reliability | 15/15 | shared mixer/audio dependency injection; engine-owned and shared lifecycle separated; server references never replaced |
| Operator UX | 19/20 | active TypeScript Soundcheck cockpit, real backend/channel state, start/stop/status, stages, blockers and proposal count |
| Verification and truth | 15/15 | focused tests, frontend tests/build/typecheck, no-WING gate, browser QA and full backend suite |
| **Total** | **98/100** | target met |

## Deductions

- `-1`: USB Main L/R is console mix evidence, not room/PA truth. A calibrated
  room microphone and operator SPL/feedback validation remain external.
- `-1`: the cockpit currently runs the canonical all-channel profile; custom
  per-stage timing and operator channel-set presets are not yet editable there.

## Implemented

- `AutoSoundcheckEngine.run()` now performs measure -> derive -> review in
  observation mode instead of returning an empty proposal queue.
- Recommendation ids are content-derived and upserted. Targets are explicitly
  marked `absolute_target`, never relative-only.
- Direct `auto_apply`, feedback-emergency constructor flags and CLI flags now
  fail closed and point to `apply_operator_proposal`/supervised workflows.
- Configured correction promotions select recommendation envelopes only; they
  are not copied into direct-write authority.
- Shared mixer/audio runtimes are reused without stop/disconnect, and the server
  no longer adopts engine-owned fallback objects.
- The active frontend now contains a Soundcheck route with honest runtime state,
  zero-write safety contract, channel evidence and start/stop controls.

## Validation

- Focused soundcheck/authority suite: `96 passed`.
- Frontend: `104 passed`.
- TypeScript: `npx tsc --noEmit` passed.
- Frontend production build: passed.
- no-WING quick gate: `status=passed`, `failed_checks=[]`,
  `contacts_wing=false`, `contacts_backend=false`, `live_write=false`,
  `live_authority_granted=false`.
- Full backend: `5855 passed, 1 skipped, 1 warning in 180.53s`.
- Browser QA: production entrypoint loaded the Soundcheck route, showed no JS
  errors, hid mock channels, and kept Start disabled without mixer/audio proof.
- Screenshot: `reports/frontend/soundcheck-cockpit-20260711.png`.

## Remaining external gate

Live concert readiness remains `WAITING_FOR_EXTERNAL_VALIDATION`. The next safe
sequence is operator-present M005 disconnect/timeout/stale-state proof followed
by M006 owner readiness review. Until then, release-ready live correction
families, live-allowed safety rows and current supervised live-write kinds remain
zero.

