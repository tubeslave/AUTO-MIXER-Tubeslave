# Brief: WING MVP Handoff Prestart Plan Fields

Date: 2026-06-14
Task ID: wing-mvp-handoff-prestart-plan-fields-20260614

## Context

The WING read-only Current First Command requires the no-WING handoff boundary
to show backend-port required-false plan values and match verdicts before any
supervised proof can be planned. The no-WING prestart detail already records
those fields, but the handoff packet builder should also carry them directly.

## Goal

Expose `plan_<field>: false` and `<field>_matches_plan: true` for every
backend-port required-false denial field in the WING read-only handoff packet
and rendered packet markdown.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused handoff packet and no-WING release checks.
- Project Director/WING rails sanity check.
- Compile and whitespace diff.
