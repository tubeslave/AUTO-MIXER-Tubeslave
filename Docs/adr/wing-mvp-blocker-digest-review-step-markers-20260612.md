# ADR: WING MVP Blocker Digest Review Step Markers

## Context

The WING read-only preflight runbook requires
`operator_sequence_mvp_blocker_anchor_digest_review_step` to include
`mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and `current first open
blocker`. The no-WING handoff validator exposes this step, but the first blocker
selected action only required the boolean
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`.

## Options considered

1. Keep the marker requirement only in runbook/validator surfaces.
2. Promote the marker requirement into selected action and autoloop docs.
3. Add a new runtime gate.

## Decision

Promote the digest-review step marker requirement into the first WING MVP
blocker selected action, generated progress, and autoloop docs.

## Why this won

Project Director and the autoloop prompt are the next-cycle entrypoints. They
should carry the same operator-visible digest-review condition that the handoff
validator enforces.

## Rejected alternatives

Keeping the markers only in the runbook leaves the selected action less
specific than the proof handoff. A new runtime gate would duplicate existing
handoff validation.

## Implementation plan

- Update blocker table/current command.
- Regenerate MVP progress.
- Update autoloop prompt/README.
- Add focused assertions.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- Full pytest before commit.

## Risks and rollback

This change does not contact WING and does not grant live authority. Rollback is
to revert the text, regenerated progress, focused tests, and report/index entry.
