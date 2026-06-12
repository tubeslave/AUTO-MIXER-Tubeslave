# ADR: WING Read-Only Handoff Anchor Digest

Date: 2026-06-12

## Context

The WING read-only handoff packet now carries `mvp_blocker_anchor` for the
current first open MVP blocker. The no-WING validator compares the packet anchor
to `.ai/project-layer/wing_mvp_blockers.md`, but the operator-visible summaries
did not include a compact fingerprint for the exact blocker-row context.

## Options considered

1. Keep object comparison only.
2. Hash the raw Markdown blocker row.
3. Hash the canonical JSON anchor object.

## Decision

Use option 3. The handoff packet now includes `mvp_blocker_anchor_sha256`, and
no-WING validation fails closed unless that digest matches both the packet
anchor and the current first-open blocker anchor.

## Why this won

Canonical JSON avoids Markdown formatting sensitivity while preserving a compact
reviewer-visible fingerprint. The digest supports auditability without
contacting WING or widening live authority.

## Rejected alternatives

- Object comparison only: correct but less convenient in summaries.
- Raw Markdown hash: too sensitive to table formatting.

## Implementation plan

- Add canonical JSON SHA256 helper in no-WING tooling.
- Add digest field to handoff packet builder.
- Validate digest against packet and current blocker anchor.
- Render digest in packet and no-WING summaries.
- Add valid and mismatch tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This is an offline validation change only. Rollback removes the digest field,
validation checks, tests, and documentation. No backend session, WING
connection, or OSC write path is affected.
