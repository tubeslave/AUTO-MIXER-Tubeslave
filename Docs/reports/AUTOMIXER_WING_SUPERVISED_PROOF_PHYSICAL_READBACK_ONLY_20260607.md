# Automixer WING Supervised Proof Physical Readback Only - 2026-06-07

## Summary

This slice tightens the physical WING proof runner so `--execute` can pass only
with physical readback evidence. Cache readback remains acceptable for dry-run
planning, but not for physical supervised proof.

## What changed

- `tools/wing_operator_proposal_validation_runner.py` now forces physical
  pre-read for `--execute`.
- Apply and rollback success require `verified_from_physical_readback`.
- Proof summaries expose whether physical readback was required.
- `tools/no_wing_release_check.py` includes a static runner-policy gate to keep
  cache out of physical proof semantics.
- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` now lists physical
  pre-read/apply/rollback readback as required expected evidence.

## Safety boundary

No WING hardware was contacted. This does not grant live authority, does not
close readiness rows, and does not widen writes beyond the existing supervised
fader/gain proof path.

## Validation

- `python3 -m py_compile tools/wing_operator_proposal_validation_runner.py tools/no_wing_release_check.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_operator_proposal_validation_runner.py tests/test_no_wing_release_check.py`
  - `196 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_physical_readback_only_20260607`
  - PASS, top-level `status: passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
  - PASS, `on_target`, score `90`
- `python3 tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3025 passed, 1 skipped, 1 warning`
