# ADR: Safety kernel trace binding

## Context

M020 introduced a focused safety-kernel trace contract, but M013 integration
bundle promotion still trusted the loose `safety_kernel_trace_bound`
integration check. That left an integration-level bypass: a bundle could mark
the trace as bound without carrying deterministic allow/reject scenario,
guard-check and audit-field evidence.

## Options Considered

1. Keep accepting `safety_kernel_trace_bound` as a sufficient integration
   check.
2. Require `safety_kernel_trace_manifest` to pass the M020 contract before
   offline integration promotion.
3. Treat passing replay, fault-recovery and integration-policy evidence as
   sufficient proof that the safety kernel was exercised.

## Decision

Use option 2. `backend/evaluation_integration_bundle_contract.py` now consumes
the M020 safety-kernel trace contract result before reporting offline
integration promotion readiness.

## Why This Won

The binding makes non-bypassable safety-kernel evidence explicit at the same
composition layer that already binds replay, fault recovery and integration
policy. Consumers see deterministic trace failures through a
`safety_kernel_trace:` child prefix instead of trusting a loose boolean.

## Rejected Alternatives

The loose check was rejected because it can be set without trace records,
guard checks or audit fields. Inferring safety-kernel coverage from other
child evidence was rejected because those contracts prove scenario outcomes,
not that each candidate action flowed through deterministic safety checks.

## Implementation Plan

1. Evaluate `safety_kernel_trace_manifest` with
   `evaluate_safety_kernel_trace_contract`.
2. Keep `safety_kernel_trace_bound` as an integration-check summary only.
3. Block promotion with `safety_kernel_trace_contract_not_ready` unless the
   M020 result is ready for safety-kernel trace use.
4. Prefix safety-kernel trace blockers for integration bundle consumers.
5. The integration contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_safety_kernel_trace_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m021_safety_kernel_trace_binding_requires_m020_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: existing integration bundles that only set `safety_kernel_trace_bound`
will now fail closed. Mitigation: the blocker explicitly names
`legacy_safety_kernel_trace_bound_without_contract_manifest` and points callers
to the M020 manifest shape.

Rollback: revert the M021 commit. This restores the loose trace-bound behavior
without touching live transport or runtime mixer behavior.
