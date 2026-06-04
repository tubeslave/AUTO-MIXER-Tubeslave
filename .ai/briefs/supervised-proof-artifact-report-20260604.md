# Supervised Proof Artifact Report Brief

Task id: supervised-proof-artifact-report-20260604
Date: 2026-06-04

## Context

The proof artifact evaluator can judge one normalized or runner-style proof
summary, but the project also needs an operational report gate that an operator
or CI job can run after dry-run or WING validation. Without that, proof evidence
stays scattered in artifact directories and promotion readiness remains hard to
audit.

## Objective

Add an offline report tool that discovers local `summary.json` files, evaluates
supervised fader/gain proof artifacts, ignores unrelated summaries, and writes a
machine-readable report separating dry-run readiness, physical supervised
candidates, blocked artifacts, and invalid summaries.

## Safety Boundary

- Read local JSON artifacts only.
- Do not import mixer transports.
- Do not connect to backend, WebSocket, OSC, WING, or dLive.
- Do not create or apply console actions.
- Do not mark Full Autonomous ready.

## Expected Files

- `tools/evaluate_supervised_proof_artifacts.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/proposals/codex-supervised-proof-artifact-report-20260604.md`
- `Docs/adr/supervised-proof-artifact-report-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused report/evaluator/no-WING tests.
- Run broader readiness/registry/write-boundary tests.
- Run the standard full pytest command.
