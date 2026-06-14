# ADR: WING MVP Readiness Prestart Summary Policy Bridge

## Context

The WING read-only connection proof remains the first open MVP blocker.
Accepted candidates must bind to a passed no-WING prestart summary with
`backend_port_prestart_summary_type: no_wing_release_check` and
`backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes`.

The autonomy promotion readiness bridge already surfaced prestart summary path,
SHA, status, and no-contact flags, but did not explicitly expose the type or
connection-policy match against the shared contract.

## Options considered

1. Keep the readiness bridge unchanged and rely on review/proof-plan validation.
2. Surface the type and connection-policy values plus contract-match booleans in
   the readiness bridge.
3. Refactor readiness bridge output around a shared typed prestart summary
   contract.

## Decision

Expose the prestart summary type and connection-policy contract matches in
`backend/autonomy_promotion_readiness.py`, and cover them in focused readiness
tests.

## Why this won

It makes manual readiness review see the same no-WING prestart contract that
gates accepted candidates and supervised proof planning, without granting live
authority or closing readiness from candidate evidence.

## Rejected alternatives

Review/proof-plan-only validation leaves the readiness bridge less explicit
than the rest of the WING contract. A typed refactor would reduce duplication
but is broader than this blocker-first slice.

## Implementation plan

- Import the shared prestart summary type and connection-policy constants.
- Add report-only type/policy fields and contract-match booleans to the WING
  read-only evidence summary in autonomy promotion readiness.
- Extend the focused readiness test.
- Record validation in a report and link it from `Docs/INDEX.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_autonomy_promotion_readiness.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_autonomy_promotion_readiness.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/autonomy_promotion_readiness.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to report-only snapshot shape. Rollback removes the new fields
and assertions. No live WING control, backend startup, console query, readiness
closure, or live authority is introduced.
