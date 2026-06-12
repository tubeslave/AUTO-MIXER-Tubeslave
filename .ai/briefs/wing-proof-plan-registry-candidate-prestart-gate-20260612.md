# WING Proof Plan Registry Candidate Prestart Gate

## Context

The supervised WING proof plan now requires accepted read-only preflight source ids to match candidate review rows carrying the backend-port prestart contract. The processing registry still described the method as a generic operator checklist.

## Task

Update processing registry metadata so the method inventory exposes the candidate-prestart gate, blocker, and non-executor status.

## Constraints

- Registry/test/docs only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- Keep `supervised_wing_proof_plan` shadow-only and outside live executor profiles.

## Test Plan

- Focused processing registry test.
- Affected registry/no-WING/WING proof-plan tests.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
