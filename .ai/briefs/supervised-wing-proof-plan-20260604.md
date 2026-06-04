# Supervised WING Proof Plan Brief

Task id: supervised-wing-proof-plan-20260604
Date: 2026-06-04

## Context

The current proof artifact report shows no physical promotion candidates for
fader or gain. The next useful step is to create a reproducible operator
checklist for the bounded physical WING proof runs that would produce those
artifacts.

## Objective

Add a read-only WING proof plan generator that turns promotion evidence rows
into explicit operator proof steps with prerequisites, abort conditions,
expected artifacts, and non-executing command tuples for the existing physical
proof runner.

## Safety Boundary

- Do not connect to WING, backend, WebSocket, OSC, or dLive.
- Do not execute generated commands.
- Only plan negative dB proof moves.
- Require explicit operator approval.
- Keep `execution_allowed_by_plan=false`, `live_write=false`, and
  `live_widening_allowed=false`.

## Expected Files

- `backend/supervised_wing_proof_plan.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- `Docs/adr/supervised-wing-proof-plan-20260604.md`

## Test Plan

- Compile touched Python files.
- Run proof plan, proof report, evidence bridge, registry, and no-WING tests.
- Smoke-check current artifacts produce proof-required plan steps.
- Run the standard full pytest command.
