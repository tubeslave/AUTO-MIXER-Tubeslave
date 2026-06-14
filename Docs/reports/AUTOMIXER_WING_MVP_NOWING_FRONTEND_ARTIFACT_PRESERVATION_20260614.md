# Automixer WING MVP no-WING Frontend Artifact Preservation

Date: 2026-06-14
Status: Validated

## Summary

The no-WING release checker now lets individual command checks declare tracked
repo artifacts that must be preserved across subprocess execution. The frontend
e2e check preserves `frontend/test-results/.last-run.json`, so Playwright test
metadata does not leave unrelated dirty state after no-WING validation cycles.

## Why

Goal-mode WING work depends on clean, repeatable no-WING validation before any
real hardware proof attempt. A tracked frontend artifact that flips during
validation creates noisy diffs and can obscure the real WING blocker state.

## Changed surfaces

- `tools/no_wing_release_check.py` adds `CommandCheck.preserve_paths`, snapshots
  those files before subprocess execution, restores changed files afterward,
  and removes preserved files that were absent before the command.
- `tools/no_wing_release_check.py` validates preserved paths stay inside the
  Automixer checkout.
- `frontend_e2e` preserves `frontend/test-results/.last-run.json`.
- `tests/test_no_wing_release_check.py` covers preserved existing-file restore,
  created-file removal, and preserve-path repo-boundary rejection.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from documentation or offline evidence.

## Validation

- Focused changed-surface tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short
  -x`: 319 passed in 3.80s.
- no-WING quick with frontend:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --continue-on-failure
  --command-timeout-sec 120 --artifact-root
  /tmp/automixer_goal_frontend_preserve_nowing`: exited 1 fail-closed with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, and failed checks `backend_port_free`,
  `wing_readonly_preflight_plan`, and
  `wing_readonly_preflight_handoff_packet_artifact_schema`.
- The same no-WING summary reported `frontend_e2e: passed`, and
  `git status --short -- frontend/test-results/.last-run.json` was clean after
  the run.
- `/opt/homebrew/bin/git diff --check`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- Full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3314 passed, 1 skipped, 1 warning in 168.52s.
