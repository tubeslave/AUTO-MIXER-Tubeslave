# Supervised Proof Remediation Plan Brief

Task id: supervised-proof-remediation-plan-20260604
Date: 2026-06-04

## Context

The proof artifact report can now fail closed on legacy WING summaries and
blocked proof rows. That is safe, but not yet actionable enough: an operator
still needs a precise list of which artifacts require rerun, cleanup, or manual
review before supervised promotion can move forward.

## Objective

Add a read-only remediation plan that turns blocked proof report rows into
explicit follow-up actions without executing any WING command or changing live
policy.

## Safety Boundary

- No live WING run in this task.
- No execution of remediation commands.
- No automatic safety-row update.
- No Full Autonomous or live write widening.

## Expected Files

- `backend/supervised_proof_remediation_plan.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- `tests/test_supervised_proof_remediation_plan.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_processing_registry.py`
- `Docs/adr/supervised-proof-remediation-plan-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused remediation/report/registry tests.
- Run broader supervised proof/readiness/no-WING tests.
- Run the standard full pytest command.
