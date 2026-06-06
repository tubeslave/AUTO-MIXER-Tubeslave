# Automixer WING Read-Only Preflight Operator Confirmation - 2026-06-07

## Summary

This slice hardens the first WING MVP blocker by requiring a machine-readable
operator channel confirmation for physical readback preflight runs.

`tools/wing_readonly_preflight_runner.py` now requires
`--operator-confirm-channels` to exactly match `--channels` whenever
`--require-physical-readback` is used. The preflight summary records the
confirmation, and evidence intake rejects artifacts with missing or mismatched
confirmation. The offline review tool also surfaces confirmed channels,
`summary_path`, and `events_path` in its artifact details so manual review can
verify the operator-approved scope and inspect the exact captured files.
Evidence intake also rejects artifacts whose readback rows do not exactly match
the declared `channels` by `value_types` target matrix.

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- No write, proposal, operator-mode mutation, rollback, emergency-stop, connect,
  or disconnect request was added.
- The change only makes read-only proof artifacts stricter.
- A clean artifact still requires manual review and does not grant live
  readiness.

## Behavior

Valid physical preflight command shape:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight
```

Fail-closed cases:

- `--require-physical-readback` without `--operator-confirm-channels`
- malformed confirmation values
- confirmation channel list that differs from `--channels`
- no-WING release-check capture template whose `--operator-confirm-channels`
  value differs from its `--channels` value
- summary missing `operator_channel_confirmation_required`
- summary missing or mismatching `operator_confirmed_channels`
- readback rows for channels or value types outside the declared preflight
  target matrix
- review report missing operator-confirmed channel or artifact path details for accepted
  artifacts

## Files Changed

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`
- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/adr/wing-readonly-preflight-operator-channel-confirmation-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py tools/review_wing_readonly_preflight_artifacts.py tools/no_wing_release_check.py tools/codex_autoloop/project_director_agent.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_no_wing_release_check.py` - `173 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_no_wing_release_check.py` - `152 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_no_wing_release_check.py tests/test_autonomy_promotion_readiness.py` - `164 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py` - `165 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py` - `23 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-operator-confirmation --command-timeout-sec 300` - PASS
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-confirmation-match --command-timeout-sec 300` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2899 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/git diff --check` - PASS

## Remaining WING MVP Blocker

The first blocker remains open. The operator still must run the updated
read-only preflight command against the intended backend/WING session and then
review the captured `summary.json` plus `events.jsonl`.
