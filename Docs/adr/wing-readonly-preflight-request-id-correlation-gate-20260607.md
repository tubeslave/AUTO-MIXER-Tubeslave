# ADR: WING Read-Only Preflight Request-ID Correlation Gate - 2026-06-07

## 1. Context

The supervised WING Rack MVP requires a real read-only WING preflight artifact
before supervised fader/gain proof. Existing tooling checked physical readback,
safe request types, event ordering, and channel/value/address identity. It did
not require an explicit response correlation id.

## 2. Options considered

1. Keep relying on strict event order.
2. Add request ids only to runner summaries.
3. Add request ids to backend read-only responses, runner checks, and offline
   artifact validation.

## 3. Decision

Use option 3.

## 4. Why this won

The proof artifact should demonstrate that each readback came from the exact
query sent by the runner. Backend echo plus event-log validation gives stronger
evidence than event order alone while keeping the path read-only.

## 5. Rejected alternatives

Strict order alone cannot distinguish a stale matching response from a response
to the current query. Summary-only request ids would not prove the backend
response carried the correlation field.

## 6. Implementation plan

- Generate deterministic request ids for read-only fader/gain queries.
- Echo request ids from `query_mixer_parameter` success and blocked responses.
- Add runner blockers for response request-id mismatch.
- Require `request_id_policy` and request-id matching in offline evidence.
- Update no-WING plan validation and runbook wording.

## 7. Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/wing_readonly_preflight_evidence.py backend/handlers/mixer_handlers.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_no_wing_release_check.py tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_reads_gain_without_writing tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_blocks_unsupported_address`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_request_id_correlation_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

The main risk is rejecting older preflight artifacts that lack request ids.
Rollback is to revert the request-id policy checks and preserve the existing
channel/value/address identity gates.
