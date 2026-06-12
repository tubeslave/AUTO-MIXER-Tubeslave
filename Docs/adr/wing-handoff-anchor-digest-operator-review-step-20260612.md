# ADR: WING Handoff Anchor Digest Operator Review Step

Date: 2026-06-12

## Context

WING read-only handoff packets already include `mvp_blocker_anchor` and
`mvp_blocker_anchor_sha256`, and no-WING validation rejects stale or mismatched
digests. The packet operator sequence is the checklist an operator follows
before supervised proof planning, so it should explicitly include the digest
review instead of relying only on summary fields.

## Options considered

1. Keep digest validation only in packet fields and release summaries.
2. Add a runbook note only.
3. Add an operator-sequence step and fail closed when it is missing.

## Decision

Use option 3. The generated handoff packet now includes an operator-sequence
step requiring review of `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
the current first open blocker. The no-WING validator fails closed if that step
is absent.

## Why this won

The operator sequence is the actionable handoff checklist. Placing the digest
review there makes stale blocker-anchor detection visible before the final
supervised prestart gate.

## Rejected alternatives

- Summary-only validation: correct but too easy to miss during manual review.
- Runbook-only prose: not machine-enforced and does not reject stale generated
  packets.

## Implementation plan

- Update the handoff packet builder operator sequence.
- Add no-WING validation and summary fields for the review step.
- Update tests and runbook documentation.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py tests/test_codex_direction_guard.py`
- Generate and validate a temporary WING read-only handoff packet.
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This is an offline handoff and validation change only. Rollback removes the
operator-sequence digest review step, summary fields, and tests. No WING,
backend, or OSC behavior changes.
