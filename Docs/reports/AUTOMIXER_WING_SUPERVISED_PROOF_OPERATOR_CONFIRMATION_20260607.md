# Automixer WING Supervised Proof Operator Confirmation - 2026-06-07

## Summary

This slice makes the physical supervised WING proof runner require a
machine-checkable operator confirmation before any `--execute` run can open a
WebSocket session. The confirmed channel, value type, and delta must match the
requested action exactly.

## What changed

- `tools/wing_operator_proposal_validation_runner.py` now requires
  `--operator-confirm-channel`, `--operator-confirm-value-type`, and
  `--operator-confirm-delta-db` for physical `--execute`.
- `backend/supervised_wing_proof_plan.py` adds those flags to generated physical
  proof commands.
- `tools/no_wing_release_check.py` rejects next WING proof plans whose command
  confirmation flags are missing or stale.
- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` shows the required flags.

## Safety boundary

This does not run WING hardware, does not grant live authority, does not mark
the read-only preflight blocker closed, and does not widen writes beyond the
existing supervised fader/gain proof path.

## Validation

- `python3 -m py_compile tools/wing_operator_proposal_validation_runner.py backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_operator_proposal_validation_runner.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py`
  - `201 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_operator_confirmation_20260607`
  - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
  - PASS, `on_target`, score `90`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3021 passed, 1 skipped, 1 warning`
