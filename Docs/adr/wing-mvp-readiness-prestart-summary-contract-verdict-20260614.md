# ADR: WING MVP Readiness Prestart Summary Contract Verdict

## Context

The WING read-only connection proof remains the first open MVP blocker.
Autonomy promotion readiness surfaces the accepted-candidate prestart summary
fields, but manual readiness review needs a single fail-closed verdict and
violation list for the full prestart summary contract.

The authoritative validator already exists as
`wing_readonly_preflight_candidate_prestart_contract_violations`.

## Options considered

1. Keep only individual field matches in the readiness snapshot.
2. Use the shared validator to emit a contract verdict and violations list.
3. Duplicate the prestart contract checks inside readiness.

## Decision

Use the shared validator and expose
`prestart_backend_port_summary_contract_valid` plus
`prestart_backend_port_summary_contract_violations` in the readiness overlay.

## Why this won

It gives operators and agents a clear contract verdict while preserving the
existing no-live, no-readiness-closure boundary. Reusing the shared validator
keeps readiness aligned with review decisions and supervised proof planning.

## Rejected alternatives

Individual field matches are useful but do not give a single verdict.
Duplicating checks risks drift from the WING prestart contract.

## Implementation plan

- Import the shared prestart contract validator in
  `backend/autonomy_promotion_readiness.py`.
- Emit contract-valid and violation-list fields.
- Emit a fail-closed `no_valid_readonly_preflight_candidate` violation when no
  valid candidate exists.
- Extend focused readiness tests for valid and missing-candidate paths.
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

Risk is limited to report-only readiness snapshot shape. Rollback removes the
new fields and assertions. No live WING control, backend startup, console
query, readiness closure, or live authority is introduced.
