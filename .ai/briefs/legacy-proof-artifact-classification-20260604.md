# Legacy Proof Artifact Classification Brief

Task id: legacy-proof-artifact-classification-20260604
Date: 2026-06-04

## Context

The supervised proof report can read historical WING operator proposal
summaries. Some old executed summaries contain `execute=true` and physical
readback fields, but were written before the explicit proof schema added
`proof_mode` and `physical_wing`.

## Objective

Classify old executed WING proof-like summaries without explicit proof schema
as blocked legacy evidence, not as dry-run-ready artifacts.

## Safety Boundary

- No live WING run in this task.
- No execution of generated proof commands.
- No live write policy widening.
- No retroactive promotion of historical artifacts.

## Expected Files

- `backend/supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_report.py`
- `Docs/adr/legacy-proof-artifact-classification-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused evaluator and report tests.
- Run broader supervised proof/readiness/no-WING tests.
- Run the standard full pytest command.
