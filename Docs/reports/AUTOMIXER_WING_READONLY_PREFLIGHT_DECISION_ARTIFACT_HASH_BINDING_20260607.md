# Automixer WING Read-Only Preflight Decision Artifact Hash Binding - 2026-06-07

## Summary

This slice strengthens the first supervised WING MVP blocker by binding operator
review decisions to the exact bytes of the reviewed preflight artifacts.

File-backed read-only preflight evidence now exposes `summary_sha256` and
`events_sha256`. Pending review decision templates include those fields, and the
decision ledger rejects decisions whose hashes are missing or stale.

## Safety Boundary

- No live WING/OSC/WebSocket action was run.
- No write, proposal, rollback, emergency-stop, connect, or disconnect request
  was added.
- The change only strengthens offline artifact review and decision validation.
- A clean decision still does not grant live authority or close readiness rows.

## Behavior

Accepted or rejected decision records must now include:

```json
{
  "summary_sha256": "<sha256 of summary.json>",
  "events_sha256": "<sha256 of events.jsonl>"
}
```

Fail-closed cases:

- Decision missing `summary_sha256` for a file-backed candidate
- Decision missing `events_sha256` for a file-backed candidate
- Decision `summary_sha256` does not match the current `summary.json`
- Decision `events_sha256` does not match the current `events.jsonl`
- Review tool cannot build a decision template because candidate digest
  identity is incomplete

## Files Changed

- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-decision-artifact-hash-binding-20260607.md`
- `.ai/briefs/wing-readonly-preflight-decision-artifact-hash-binding-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-decision-artifact-hash-binding-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_ARTIFACT_HASH_BINDING_20260607.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Validation

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py backend/wing_readonly_preflight_review_decision.py tools/review_wing_readonly_preflight_artifacts.py tools/review_wing_readonly_preflight_decisions.py tools/no_wing_release_check.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py` - `52 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py` - `240 passed`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_wing_readonly_decision_hash_binding_20260607 --command-timeout-sec 300` - PASS
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS (`on_target`, score `90`)
- `PYTHONPATH=backend python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `3032 passed, 1 skipped, 1 warning`

## Remaining WING MVP Blocker

The first blocker remains open. The operator still must run the updated
read-only preflight against the intended backend/WING session, review the
captured `summary.json` and `events.jsonl`, and record exactly one accepted
decision record before supervised proof planning can proceed.
