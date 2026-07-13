# Brief: full project readiness audit to 95

Date: 2026-07-11

## Objective

Audit the current dirty checkout as a live-sound product, repair verified defects
in priority order, and raise the evidence-backed project readiness score to at
least 95/100 without inventing hardware, owner, listening, or release evidence.

## Required outcome

- One blocker-first score covering safety, runtime, product UX, verification,
  documentation truth, and release evidence.
- P0/P1/P2 findings tied to files, tests, and artifacts.
- Regression tests for every executable bug fix.
- No unattended WING contact or live write.
- No widening of normal supervised authority from code presence, unit tests, or
  self-attested proof output.
- Remaining external requirements stay explicit if they prevent 95/100.

## Current baseline

- Branch: `codex/automixer-autopilot-control-plane`.
- Worktree: dirty with active M005/M006, soundcheck, frontend, test, docs, and
  generated-artifact work that must be preserved.
- Full backend suite: `5889 passed, 1 skipped, 1 warning`.
- Frontend: TypeScript passed, `104 passed`, production build passed.
- No-WING quick gate passed, but generated a new artifact bundle.
- Lint, dead-code, and shellcheck tools are not installed, so those categories
  are unmeasured rather than green.

## Verified initial P0 findings

1. `supervised_console_parameter_registry.py` hardcodes normal live authority
   for write kinds whose artifacts still deny live authority and whose owner
   readiness decision is absent.
2. `wing_supervised_family_proof_runner.py` can contact WING and execute a live
   proof without an explicit execute flag or typed operator confirmation, and
   self-populates its approval fields.
3. Unit tests append synthetic events to the mutable canonical manual-write
   evidence log; the M006 report cites that same log as release evidence.
4. Multi-address apply and exception paths can leave partial writes outside the
   coordinator rollback set and without a guaranteed emergency stop.

## Safety boundary

All work in this audit is offline/local unless a future operator-present step is
explicitly requested. The repair target is fail-closed software and honest
evidence. It is not permission to operate the console.

## Initial scoring rule

The score is capped below 95 while any P0 live-authority bypass, mutable proof
contamination, or unresolved critical rollback gap exists. Green unit tests do
not override that cap.
