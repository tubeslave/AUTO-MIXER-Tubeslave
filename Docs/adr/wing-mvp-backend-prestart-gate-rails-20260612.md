# ADR: WING MVP Backend Prestart Gate Rails

## Context

The WING read-only handoff packet now carries structured
`backend_port_prestart_gate` and `expected_backend_port_prestart_gate` objects.
The no-WING validator fails closed if those objects drift. The top-level MVP doc
and autonomous cycle prompt still referenced only the final supervised prestart
gate.

## Options considered

1. Keep first-gate requirements only in the runbook and validator.
2. Add first-gate wording to the MVP doc only.
3. Add first-gate wording to the MVP doc, Codex prompt, docs index, and rails
   tests.

## Decision

Use option 3. The structured backend-port prestart gate is now part of the
top-level WING MVP rails and autonomous cycle prompt.

## Why this won

The first prestart gate is the safety boundary before any intended
backend/WING session. Future cycles should not plan supervised proof unless the
same structured first-gate evidence appears in the no-WING handoff summary.

## Rejected alternatives

- Runbook-only wording was rejected because goal-mode task selection starts
  from the MVP doc and cycle prompt.
- MVP-doc-only wording was rejected because autonomous prompt drift would still
  be possible.

## Implementation plan

- Update MVP stop-the-line rules and current-next-action section.
- Update the Codex cycle prompt's WING Rack order and pre-supervised checks.
- Link the structured prestart gate report in the docs index.
- Extend WING MVP rails tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- Full repository pytest.

## Risks and rollback

This change can make future rails tests stricter. Roll back by removing the
first-gate wording from the MVP rails, but that would reintroduce a mismatch
between top-level guidance and handoff validation.
