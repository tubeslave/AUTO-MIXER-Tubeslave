# ADR: WING MVP Evidence Prestart Detail Contract

## Context

The first open WING MVP blocker remains `WING read-only connection proof`. The
real read-only runner validates the no-WING prestart summary before WebSocket.
The post-run evidence intake also reads the referenced prestart summary while
deciding whether a captured artifact can become a read-only preflight candidate.
Before this change, it checked no-contact metadata but did not enforce the full
backend-port detail contract inside that referenced prestart file.

## Options considered

1. Trust runner validation only.
2. Document the reviewer expectation without code.
3. Enforce the prestart detail list and required-false detail fields in evidence
   intake.

## Decision

Use option 3.

## Why this won

Artifact review is a separate gate. It must reject stale or edited prestart
summary files even if a runner once produced the captured artifact. Enforcing the
same detail contract in review keeps offline evidence from becoming stronger
than its referenced no-WING prestart boundary.

## Rejected alternatives

Runner-only validation does not protect post-run artifact replacement, manual
file edits, or stale reviewed evidence. Documentation-only coverage cannot
protect accepted candidate selection.

## Implementation plan

- Add the shared required-false prestart detail field set to evidence intake.
- Validate the referenced no-WING prestart backend check detail list.
- Validate every required-false detail field and compare recorded summary values
  against the referenced prestart summary.
- Add focused regression tests and runbook wording.
- Regenerate latest MVP progress.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- Standard full pytest.

## Risks and rollback

This is offline artifact validation. It does not contact WING, stop listeners,
or grant live authority. Roll back by removing the evidence checks, tests,
runbook wording, report/index entries, and regenerated progress report.
