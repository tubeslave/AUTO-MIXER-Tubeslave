# ADR: Fault-Injection Recovery Contract

## Context

The Project Completion Gate requires simulator, fault-injection, recovery,
rollback, emergency-stop and audit evidence before Automixer can be considered
ready for higher autonomy. Existing no-WING and replay artifacts are useful,
but they must not be confused with live authority, readiness closure or real
hardware proof.

## Options Considered

1. Keep recovery and fault-injection checks as narrative roadmap text.
2. Add a fail-closed simulator fault recovery contract.
3. Wire simulator fault evidence directly into live readiness promotion.

## Decision

Add `backend/fault_injection_recovery_contract.py` as a deterministic,
offline-only contract. It validates required failure modes, simulator /
fault-injection / recovery evidence, recovery outcomes, audit fields and an
approved recovery policy before reporting `ready_for_fault_recovery_promotion`.

The contract always returns `live_authority_granted: false` and
`readiness_closure_allowed: false`.

## Why This Won

This gives the project a concrete recovery/failure gate that can advance
without WING hardware while preserving the hard boundary between simulator
evidence and supervised live proof. It also makes rollback, emergency-stop,
safety-kernel and audit evidence explicit enough to test.

## Rejected Alternatives

- Narrative-only tracking was rejected because it cannot be used as a stable
  regression gate.
- Direct live readiness promotion was rejected because simulator fault evidence
  is not hardware proof and must not grant authority.

## Implementation Plan

1. Add the simulator fault recovery contract module.
2. Add unit tests for incomplete evidence, complete simulator evidence, live or
   hardware claims, missing recovery/audit evidence and forbidden live transport
   imports.
3. Bind the contract to `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`,
   `Docs/CONSOLE_CAPABILITY_MATRIX.md`, `Docs/CURRENT_STATE.md`,
   `ops/autopilot/LAST_REPORT.md` and the M012 rail.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fault_injection_recovery_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

The main risk is treating complete simulator fault evidence as stronger than it
is. The contract mitigates this by keeping live authority and readiness closure
false even when fault recovery promotion is ready.

Rollback is to revert the M012 commit. No live console path imports or calls
this contract.
