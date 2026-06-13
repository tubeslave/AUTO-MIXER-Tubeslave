# WING MVP Real Artifact Expected-Artifacts Binding

## Summary

The WING read-only evidence review now binds accepted file-backed candidates to
the offline plan's `expected_artifacts`. A clean candidate must have
`summary_path` and `events_path` matching the plan's expanded `<run_id>`
summary/event globs.

The first WING MVP blocker remains `WING read-only connection proof`; this
report is cited in that blocker row as tooling evidence only. It does not close
the blocker without a real WING artifact and manual review.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No fader, gain, trim, EQ, compressor, routing, or scene mutation.
- No live authority granted.
- Readiness rows remain manual-review gated.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/review_wing_readonly_preflight_artifacts.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py --tb=short`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py --tb=short`: `342 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_wing_readonly_preflight_review_decision.py --tb=short`: `29 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --backend-port 65535 --artifact-root /tmp/automixer_goal_real_artifact_expected_artifacts_binding_20260613`: passed
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_codex_direction_guard.py tests/test_wing_mvp_rails.py --tb=short`: `23 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`: selected `WING read-only connection proof`, `guard_status: on_target`, `selected_evidence_file_count: 47`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`: `status: on_target`, `stop_the_line: false`, `score: 90`
- `/opt/homebrew/bin/git diff --check`: passed
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: `3281 passed, 1 skipped, 1 warning`
