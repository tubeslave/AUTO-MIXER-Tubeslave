# Automixer WING MVP Current Evidence Index Sync

Date: 2026-06-14
Status: Validated

## Summary

The current first WING MVP blocker evidence now includes every current WING MVP
2026-06-14 report indexed in `Docs/INDEX.md`, and tests derive that expected set
from the index.

## Why

Project Director uses the first blocker evidence as `selected_evidence_files`.
If `Docs/INDEX.md` points at current WING proof-boundary reports that are absent
from that evidence list, Goal-mode cycles can miss relevant context while still
appearing on target.

## Changed surfaces

- `.ai/project-layer/wing_mvp_blockers.md` now includes the missing current
  2026-06-14 WING MVP report paths.
- `tests/test_wing_mvp_rails.py` derives current WING MVP report names from
  `Docs/INDEX.md` and asserts the first blocker evidence includes all of them.
- `tests/test_project_director_agent.py` derives current WING MVP report paths
  from `Docs/INDEX.md` and asserts Project Director selected evidence includes
  all of them.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No MVP blocker closed from documentation or offline evidence.

## Validation

- Focused WING rails and Project Director tests:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
  --tb=short -x`: 11 passed in 0.39s.
- Project Director JSON check selected `WING read-only connection proof`,
  reported `guard_status: on_target`, `guard_score: 90`,
  `guard_stop_the_line: false`, baseline coverage passed with 99 markers,
  exposed `selected_evidence_file_count: 97`, and found 46 indexed current
  reports with `missing_indexed_current_reports: []`.
- Direction guard JSON check reported `status: on_target`, `score: 90`, and
  `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/codex_autoloop/project_director_agent.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --artifact-root
  /tmp/automixer_current_evidence_index_sync_nowing`: exited 1 fail-closed on
  existing listener `127.0.0.1:8765`, with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, and no live authority.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3310 passed, 1 skipped, 1 warning in 174.74s (0:02:54).
