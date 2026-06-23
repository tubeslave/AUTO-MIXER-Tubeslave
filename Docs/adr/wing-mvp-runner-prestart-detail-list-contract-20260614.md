# ADR: WING MVP Runner Prestart Detail List Contract

## Context

The first open WING MVP blocker remains `WING read-only connection proof`. The
real `wing_readonly_preflight_runner.py` accepts a no-WING prestart summary and
an expected offline plan before opening the backend/WING WebSocket. The runner
already checks prestart status, policy, no-contact flags, backend check status,
and required false detail values. It did not separately check that the
prestart-summary backend check still declares the current
`backend_port_prestart_check_failure_detail_fields` contract.

## Options considered

1. Leave this to expected-plan mismatch validation.
2. Add documentation only.
3. Enforce the detail-field list in the runner before expected-plan/WebSocket.

## Decision

Use option 3.

## Why this won

The no-WING prestart summary is a direct input to the real read-only run. A
stale summary should fail closed even if an expected plan path is also supplied.
Checking the list in the runner keeps stale failure-detail contracts from
reaching the real backend/WING session.

## Rejected alternatives

Expected-plan-only coverage is indirect and can hide which input is stale.
Documentation-only coverage cannot protect the real runner.

## Implementation plan

- Copy `backend_port_prestart_check_failure_detail_fields` from
  `backend_port_free.detail` into prestart details.
- Compare it with the shared backend-port failure-detail field contract.
- Emit `prestart_summary_backend_port_check_failure_detail_fields_mismatch` and a
  specific operator next step on mismatch.
- Add a focused regression that proves WebSocket is not opened.
- Update the WING read-only preflight runbook and report.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_wing_readonly_preflight_runner.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- Standard full pytest.

## Risks and rollback

This is a pre-WebSocket validation change. It does not contact WING, stop
listeners, or grant live authority. Roll back by removing the runner check,
regression test, runbook wording, and council/report entries.
