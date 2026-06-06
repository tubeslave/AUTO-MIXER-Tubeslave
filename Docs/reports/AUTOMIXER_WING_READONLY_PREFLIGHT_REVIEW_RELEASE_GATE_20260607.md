# Automixer WING Read-Only Preflight Review Release Gate - 2026-06-07

## Summary

This slice connects the read-only preflight capture runner and offline artifact
review tool to the no-WING release gate. `tools/no_wing_release_check.py` now
publishes the capture command template, review command, and review exit-code
contract in its WING Read-Only Preflight Evidence summary, and it fails closed
if either CLI is missing.

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- No read-only preflight artifact was marked accepted from offline checks alone.
- No fader, gain, trim, EQ, compressor, routing, mute, scene, insert, FX, DCA,
  matrix, or autonomous write authority was added.
- Manual review of a real `wing_readonly_preflight.v1` artifact remains
  required before the first MVP blocker can close.

## Behavior

- If `tools/wing_readonly_preflight_runner.py` exists, the no-WING release
  summary includes:
  - `capture_tool_path`
  - `capture_tool_available`
  - `capture_command_template`
  - `capture_command_requires_operator_channel_confirmation`
  - `capture_command_requires_physical_readback`
- If `tools/review_wing_readonly_preflight_artifacts.py` exists, the same
  summary includes:
  - `review_tool_path`
  - `review_tool_available`
  - `review_command`
  - `review_exit_codes`
- If the capture runner is missing, the preflight evidence schema check fails
  with `wing_readonly_preflight_runner_tool_missing`.
- If the capture command template confirms a different channel list than it
  queries, the preflight evidence schema check fails with
  `wing_readonly_preflight_capture_operator_confirmation_mismatch`.
- If the review tool is missing, the preflight evidence schema check fails with
  `wing_readonly_preflight_review_tool_missing`.
- `summary.json` and `summary.md` both show the operator command:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight
```

```bash
python3 tools/review_wing_readonly_preflight_artifacts.py \
  --artifact-root artifacts/wing_readonly_preflight \
  --output Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md
```

## Files Changed

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/adr/wing-readonly-preflight-review-release-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` - `157 passed`
- `python3 -m py_compile tools/codex_autoloop/project_director_agent.py tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_no_wing_release_check.py` - `138 passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2889 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-preflight-review --command-timeout-sec 300` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Remaining WING MVP Blocker

The first blocker remains open: the operator still must run
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` against the intended backend/WING
session, capture a real summary plus `events.jsonl`, and review that artifact
before any supervised fader/gain proof.
