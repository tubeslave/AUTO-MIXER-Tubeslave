# ADR: Safety kernel trace contract

## Context

The Project Completion Gate requires that no model, LLM, classifier or policy
can bypass the deterministic Safety Kernel. Existing integration evidence only
had a loose `safety_kernel_trace_bound` check, which did not define the trace
scenarios, guard checks or audit fields required before offline integration
promotion can trust safety-kernel evidence.

## Options Considered

1. Keep `safety_kernel_trace_bound` as a loose boolean integration check.
2. Add a focused safety-kernel trace contract before binding integration
   bundles to it.
3. Treat passing replay and fault-recovery evidence as sufficient proof of
   safety-kernel enforcement.

## Decision

Use option 2. Add `backend/safety_kernel_trace_contract.py` as a fail-closed
offline contract for deterministic safety-kernel trace evidence.

## Why This Won

The focused contract makes safety-kernel evidence explicit and auditable while
preserving the existing integration bundle contract. It lets future binding
work consume a clear contract result instead of trusting a loose boolean.

## Rejected Alternatives

Loose boolean checks were rejected because they do not prove non-bypassable
guard coverage. Treating replay/fault evidence as sufficient was rejected
because those bundles prove scenario behavior, not that every candidate action
flowed through deterministic safety checks.

## Implementation Plan

1. Add `evaluate_safety_kernel_trace_contract`.
2. Require deterministic allow/reject trace scenarios.
3. Require guard checks for mode permission, allowlists, fader ceiling,
   true-peak headroom, feedback, stale-state, readback, rollback, rate limits,
   audit logging, human supervision and model-bypass blocking.
4. Require audit fields for each trace record.
5. The contract always returns `live_authority_granted: false` and
   `readiness_closure_allowed: false`.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_safety_kernel_trace_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m020_safety_kernel_trace_contract_stays_offline_and_non_bypassable -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: future integration fixtures that only set `safety_kernel_trace_bound`
will remain incomplete once M020 is bound into integration promotion. This is
intentional: they need deterministic trace records.

Rollback: revert the M020 commit. This removes the standalone trace contract
without touching live transport or runtime mixer behavior.
