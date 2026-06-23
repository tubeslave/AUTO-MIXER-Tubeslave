# Brief: WING MVP Proof Plan Required Summary Field Gate

## Task

Make no-WING validation fail closed when `next_wing_proof_plan.summary` omits any field declared by the handoff-required supervised proof-plan summary field contract.

## Context

The WING handoff packet now advertises `required_supervised_proof_plan_summary_fields`, including accepted-candidate prestart and artifact-root diagnostics. The downstream proof-plan schema validator compares many individual values, but a stale summary should also fail explicitly when any required field is missing.

## Constraints

- No live mixer transport.
- No live-write authority.
- No positive gain or fader movement.
- Reuse `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.
- Keep the WING read-only connection proof blocker open until real WING evidence exists.

## Done When

- The no-WING proof-plan schema validator exposes the required field contract.
- Missing required summary fields produce a dedicated fail-closed violation.
- Tests cover the happy path and a stale summary missing an artifact-root field.
- The decision is recorded in ADR/report docs and indexed.
