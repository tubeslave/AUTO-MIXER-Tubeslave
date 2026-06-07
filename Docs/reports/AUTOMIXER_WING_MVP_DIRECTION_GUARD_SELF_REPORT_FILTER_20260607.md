# Automixer WING MVP Direction Guard Self-Report Filter - 2026-06-07

## Summary

This slice fixes a recursive false-positive in the Codex direction guard.

Before the change, the guard could report `mixed_needs_review` while
`drift_item_count` was `0`, because it matched drift patterns inside its own
generated safety guidance or stale generated evidence tables.

After the change, drift scoring ignores:

- `## Forbidden Next Actions`
- `## Forbidden Scope`
- generated direction-guard `## Target Evidence`
- generated direction-guard `## Drift Evidence`
- generated direction-guard `## Mixed Evidence`

Target scoring still uses the full text, and real drift outside those sections
still fails the guard.

## Safety Boundary

- No WING Rack connection was opened.
- No OSC/WebSocket transport was contacted.
- No fader, gain, trim, EQ, dynamics, routing, mute, FX, scene, matrix, or DCA
  write path was widened.
- No MVP blocker was closed.
- The next real action remains the WING read-only preflight runbook when
  hardware and backend are ready.

## Files Changed

- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_codex_direction_guard.py`
- `.ai/briefs/wing-mvp-direction-guard-self-report-filter-20260607.md`
- `.ai/proposals/codex-wing-mvp-direction-guard-self-report-filter-20260607.md`
- `Docs/adr/wing-mvp-direction-guard-self-report-filter-20260607.md`
- `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SELF_REPORT_FILTER_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py` - `10 passed`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py` - PASS; status `on_target`, score `90`, `drift_item_count=0`, `mixed_item_count=0`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --output /tmp/automixer_project_director_self_report_filter.md --print` - PASS; guard status `on_target`, selected blocker `WING read-only connection proof`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py tests/test_project_director_agent.py` - `13 passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2999 passed, 1 skipped, 1 warning`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_direction_guard_self_report_filter_20260607` - PASS; top-level status `passed`

## Current Boundary

`on_target` means only that the current repository direction matches the
supervised WING MVP rail. It is not a hardware proof, not a readiness update,
and not permission to start supervised writes.
