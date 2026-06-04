# Autonomy Mode Transition Gate Brief

Task id: autonomy-mode-transition-gate-20260604
Date: 2026-06-04

## Context

The project has a four-mode runtime contract and current proof-evidence overlay
for fader/gain promotion targets. The missing layer is a read-only gate that
explains which modes are currently available, limited, or blocked based on
policy, static readiness, and proof artifacts.

## Objective

Add a read-only autonomy mode transition gate for Analysis, Assisted,
Supervised AutoMix, and Full Autonomous modes.

## Safety Boundary

- No live WING run in this task.
- No runtime operator-mode change.
- No proof command execution.
- No safety-row mutation.
- No automatic mode promotion or live write widening.

## Expected Files

- `backend/autonomy_mode_transition_gate.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- `tests/test_autonomy_mode_transition_gate.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_processing_registry.py`
- `Docs/adr/autonomy-mode-transition-gate-20260604.md`

## Test Plan

- Compile touched Python files.
- Run focused gate/report/registry tests.
- Run broader mode/proof/readiness/no-WING tests.
- Run the standard full pytest command.
