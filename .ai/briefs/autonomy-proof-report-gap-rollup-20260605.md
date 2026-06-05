# Brief: autonomy-proof-report-gap-rollup-20260605

## Context

The supervised proof artifact report already embeds the autonomy mode transition
gate. The gate can explain why Full Autonomous mode is blocked, but operators
reading the top-level report summary still have to inspect nested rows to see
the remaining readiness gap buckets.

## Goal

Expose a top-level, read-only autonomy gap rollup in the proof artifact report
summary, sourced from `autonomy_mode_transition_gate.rows[mode=auto]`.
The no-WING release gate must validate the rollup against the canonical
Supervised mode policy instead of duplicating the current live-write kinds.
The generated `summary.json` and `summary.md` should also surface the rollup so
operators can see the remaining Full Auto gaps without inspecting nested logs.

## Constraints

- Do not add live mixer reads or writes.
- Do not widen supervised live apply beyond the current fader/gain boundary.
- Keep Full Autonomous blocked until all promotion gaps have evidence.
- Preserve the existing report schema and add fields only.

## Likely files

- `backend/autonomy_mode_transition_gate.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `tools/no_wing_release_check.py`
- `backend/operator_mode_policy.py` (read-only dependency)
- `tests/test_autonomy_mode_transition_gate.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_no_wing_release_check.py`
- `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md`
- `Docs/adr/autonomy-proof-report-gap-rollup-20260605.md`

## Validation

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py tests/test_supervised_proof_artifact_report.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 -m py_compile backend/autonomy_mode_transition_gate.py tools/evaluate_supervised_proof_artifacts.py`
