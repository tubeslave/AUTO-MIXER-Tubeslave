# Brief: WING MVP Current Command Declared Denial Match Flags

Date: 2026-06-14
Task ID: wing-mvp-current-command-declared-denial-match-flags-20260614

## Context

The WING read-only handoff packet and no-WING release gate now expose declared
and recomputed plan-match verdicts for backend-port prestart denial fields.
Project Director current-command baseline coverage still only requires the
computed verdict markers.

## Goal

Require `Current First Command` and autoloop rails to carry the
`declared_<field>_matches_plan: true` markers for safe-to-stop, live-authority,
execution, readiness-closure, and backend-prestart live-authority denial
verdicts.

## Safety Boundary

- No WING hardware contact.
- No OSC/WING writes.
- No backend or frontend server lifecycle changes.
- No readiness closure.
- No live authority granted.

## Expected Validation

- Focused Project Director and WING rails tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, whitespace diff, and full test suite where the local
  environment permits.
