# Automixer WING Read-Only Preflight Session Confirmation Gate - 2026-06-07

## Summary

This slice hardens the first supervised WING MVP blocker by requiring an
explicit operator confirmation that the intended backend/WING session is ready
for physical read-only preflight.

`tools/wing_readonly_preflight_runner.py` now requires
`--operator-confirm-session-ready` whenever `--require-physical-readback` is
used. Offline plans, generated handoff packets, and real preflight summaries
record the confirmation. Evidence intake and the no-WING release gate reject
stale artifacts or command templates that omit it.

## Safety Boundary

- No live WING/OSC/WebSocket action was run for this change.
- No write, proposal, rollback, emergency-stop, connect, or disconnect request
  was added.
- The change only makes read-only proof planning and artifact intake stricter.
- A clean artifact still requires manual operator review and does not grant
  live readiness.

## Behavior

Valid physical preflight command shape:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --operator-confirm-channels 3 \
  --operator-confirm-session-ready \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight \
  --expected-plan artifacts/wing_readonly_preflight_plan/latest_plan.json
```

Fail-closed cases:

- `--require-physical-readback` without `--operator-confirm-session-ready`
- offline plan missing `operator_session_confirmation_required`
- offline plan missing `operator_confirmed_session_ready`
- physical summary missing either session-confirmation field
- no-WING capture or plan command template missing the flag
- handoff packet preview or capture command missing the flag

## Files Changed

- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_autonomy_promotion_readiness.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-session-confirmation-gate-20260607.md`
- `.ai/briefs/wing-readonly-preflight-session-confirmation-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-session-confirmation-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_CONFIRMATION_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_no_wing_release_check.py` - `228 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_autonomy_promotion_readiness.py::test_readiness_snapshot_surfaces_wing_preflight_candidate_without_closing_rows tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py` - `262 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_wing_readonly_session_confirmation_gate_20260607 --command-timeout-sec 300` - PASS
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS (`on_target`, score `90`)
- `PYTHONPATH=backend python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3031 passed, 1 skipped, 1 warning`

## Remaining WING MVP Blocker

The first blocker remains open. An operator still must run the updated read-only
preflight command against the intended backend/WING session, review the captured
`summary.json` and `events.jsonl`, and record an accepted manual review decision
before any supervised proof planning can move forward.
