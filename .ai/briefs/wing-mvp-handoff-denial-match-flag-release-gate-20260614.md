# Brief: WING MVP Handoff Denial Match Flag Release Gate

Date: 2026-06-14
Task ID: wing-mvp-handoff-denial-match-flag-release-gate-20260614

## Context

The WING read-only handoff packet now declares plan-match flags for backend-port
prestart denial fields. The no-WING release check already recomputes those
verdicts for the summary, but it must also reject stale packet declarations
when the declared flags do not match the recomputed truth.

## Goal

Make `tools/no_wing_release_check.py` fail closed when handoff packet
`*_matches_plan` declarations drift from the computed no-authority plan verdicts.

## Safety Boundary

- No WING hardware contact.
- No OSC/WING writes.
- No backend or frontend server lifecycle changes.
- No readiness closure.
- No live authority granted.

## Expected Validation

- Focused no-WING release check tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, whitespace diff, and full test suite where the local
  environment permits.
