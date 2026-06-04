# Autonomy Evidence Overlay Brief

Task id: autonomy-evidence-overlay-20260604
Date: 2026-06-04

## Context

The static autonomy readiness snapshot shows which promotion targets are
blocked by safety, read, and write coverage. The proof report now produces
promotion evidence and remediation actions, but the two views are not yet tied
together.

## Objective

Add a read-only overlay that maps current supervised proof report evidence onto
the fader/gain autonomy promotion targets.

## Safety Boundary

- No live WING run in this task.
- No proof command execution.
- No safety-row mutation.
- No automatic promotion or live write widening.
- Full Autonomous remains blocked.

## Expected Files

- `backend/autonomy_evidence_overlay.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- `tests/test_autonomy_evidence_overlay.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_processing_registry.py`
- `Docs/adr/autonomy-evidence-overlay-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused overlay/report/registry tests.
- Run broader proof/readiness/no-WING tests.
- Run the standard full pytest command.
