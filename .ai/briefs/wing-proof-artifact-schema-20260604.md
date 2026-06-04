# WING Proof Artifact Schema Brief

Task id: wing-proof-artifact-schema-20260604
Date: 2026-06-04

## Context

The proof report and WING proof plan depend on future physical runner summaries
being unambiguous. Existing `wing_operator_proposal_validation_runner.py`
summaries record `execute`, write policy, and readback evidence, but they do
not explicitly record `proof_mode` or `physical_wing`.

## Objective

Make future WING operator proposal validation artifacts explicitly identify
whether they are dry-run planning artifacts or physical proof artifacts.

## Safety Boundary

- No live WING run in this task.
- No execution of generated proof commands.
- No widening of live write policy.
- Existing legacy artifacts are not retroactively promoted.

## Expected Files

- `tools/wing_operator_proposal_validation_runner.py`
- `tests/test_wing_operator_proposal_validation_runner.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `Docs/adr/wing-proof-artifact-schema-20260604.md`

## Test Plan

- Compile touched Python files.
- Run runner/evaluator/report/proof-plan focused tests.
- Run broader readiness/write/no-WING tests.
- Run the standard full pytest command.
