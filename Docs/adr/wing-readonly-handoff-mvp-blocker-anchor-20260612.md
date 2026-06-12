# ADR: WING Read-Only Handoff MVP Blocker Anchor

Date: 2026-06-12

## Context

The supervised WING Rack MVP is advanced by the first non-closed row in
`.ai/project-layer/wing_mvp_blockers.md`. The offline read-only handoff packet
already proves it is non-live and carries structured prestart gates, but it did
not prove that it was generated for the current first open blocker row.

## Options considered

1. Keep only `blockers_not_closed_by_packet`.
2. Put the first-open blocker context in documentation only.
3. Add a machine-readable `mvp_blocker_anchor` to handoff packets and validate
   it against the current blocker table.

## Decision

Use option 3. Handoff packets now include `mvp_blocker_anchor`, and no-WING
schema validation fails closed when the anchor differs from the current
first-open blocker row.

## Why this won

The operator packet is the artifact that bridges offline preparation to a real
WING read-only proof. It must not be reusable after the MVP blocker table
changes. A table-derived anchor makes stale packets visible and machine
rejectable without contacting WING or granting live authority.

## Rejected alternatives

- `blockers_not_closed_by_packet` only: preserves the safety boundary but does
  not detect stale blocker context.
- Documentation-only anchor: useful to humans but too weak for automated
  no-WING release validation.

## Implementation plan

- Add a small blocker-table parser to `tools/no_wing_release_check.py`.
- Include the parser output in `build_wing_readonly_preflight_handoff_packet.py`.
- Render the anchor in handoff Markdown.
- Validate the packet anchor against the current table-derived anchor.
- Add tests for valid and stale anchors.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This is an offline validation change only. Rollback removes the anchor fields,
schema checks, tests, and documentation. No WING connection, backend session, or
OSC write path is affected.
