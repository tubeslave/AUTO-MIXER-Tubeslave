# ADR: WING MVP Proof Plan Canonical Prestart Helper

## Context

The WING read-only connection proof remains the first open MVP blocker. The
accepted-candidate prestart trace must prove more than the backend prestart
command: it must also bind expected artifacts, no-WING prestart summary path
and SHA, status, connection policy, and no-contact/no-live booleans.

`backend/wing_readonly_preflight_contract.py` already owns that contract. The
no-WING `next_wing_proof_plan` gate was still partially reconstructing the
check inline.

## Options considered

1. Keep the local partial validation.
2. Duplicate the full contract inline in `tools/no_wing_release_check.py`.
3. Call the shared contract helper from the release gate.

## Decision

Call the shared contract helper from `validate_supervised_proof_report_next_wing_plan`
and compare both accepted source ids and violations maps against the canonical
contract result.

## Why this won

It removes a divergent decision path and makes the release gate fail closed on
missing expected-artifact or prestart-summary trace fields before supervised
proof steps are trusted.

## Rejected alternatives

Keeping the partial check could under-report incomplete accepted candidates.
Duplicating the contract inline increases maintenance risk and makes future
contract additions easier to miss.

## Implementation plan

- Use `preflight_contract.wing_readonly_preflight_accepted_candidate_prestart_source_ids`.
- Use `preflight_contract.wing_readonly_preflight_accepted_candidate_prestart_violations_by_source`.
- Add a stale source-violations mismatch violation.
- Update tests and fixtures to carry the full accepted-candidate trace.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/no_wing_release_check.py`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_mvp_rails.py tests/test_project_director_agent.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to stricter offline release-gate validation. Rollback restores
the local partial check. No backend startup, WING connection, console query,
console write, readiness closure, or live authority is introduced.
