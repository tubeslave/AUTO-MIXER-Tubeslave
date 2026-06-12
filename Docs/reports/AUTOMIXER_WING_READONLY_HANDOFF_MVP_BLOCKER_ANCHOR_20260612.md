# Automixer WING Read-Only Handoff MVP Blocker Anchor

Date: 2026-06-12

## Summary

The WING read-only preflight handoff packet now carries a structured
`mvp_blocker_anchor` for the current first open MVP blocker. The no-WING
validator fails closed if a handoff packet anchor does not match the current
`.ai/project-layer/wing_mvp_blockers.md` first-open row.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session opened.
- No OSC/WING writes.
- No listener was stopped or reconfigured.
- No live authority granted.
- No MVP blocker was closed from documentation or offline evidence.

## Change

- Added a first-open blocker anchor helper to `tools/no_wing_release_check.py`.
- Added `mvp_blocker_anchor` to WING read-only handoff packets.
- Rendered the anchor in handoff Markdown.
- Added no-WING schema validation for current first-open blocker anchor match.
- Added stale-anchor regression coverage.
- Updated the read-only runbook to require
  `mvp_blocker_anchor_matches_current_first_open: true`.

## Why It Matters

The first WING MVP blocker is the current proof target. This change prevents a
previously generated offline handoff packet from silently passing validation
after the blocker table changes or after someone edits the packet to imply the
blocker is closed.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
  - `219 passed`
- `python3 tools/build_wing_readonly_preflight_handoff_packet.py --format json --output /tmp/wing_handoff_anchor.json --quiet`
  - `selected_blocker: WING read-only connection proof`
  - `selected_blocker_status: tooling_ready_waits_real_wing`
  - `selected_evidence_file_count: 23`
  - `packet_does_not_close_blocker: true`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  - `selected_blocker: WING read-only connection proof`
  - `selected_evidence_file_count: 23`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
  - `status: on_target`
  - `stop_the_line: false`
- `/opt/homebrew/bin/git diff --check`
  - PASS
- `python3 tools/memory/memory_lint.py`
  - PASS
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  - `3203 passed, 1 skipped, 1 warning`
