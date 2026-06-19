# ADR: Integration Evaluation Bundle Contract

## Context

The Project Completion Gate requires end-to-end soundcheck, studio and concert
scenarios to pass offline, replay, simulator, fault-injection and integration
evaluations. M011 and M012 added separate replay and simulator fault-recovery
contracts, but the project still needs a deterministic gate that proves these
evidence streams are bound together before any promotion work can treat them as
integration-level evidence.

## Options Considered

1. Keep replay and fault-recovery contracts independent and rely on roadmap
   text for integration readiness.
2. Add a fail-closed integration bundle contract that composes the child
   contracts and adds integration trace requirements.
3. Wire child contract success directly into readiness promotion.

## Decision

Add `backend/evaluation_integration_bundle_contract.py` as a deterministic,
offline-only integration gate. It evaluates the M011 replay manifest and the
M012 fault-recovery manifest, requires integration trace checks and requires an
approved integration policy before reporting
`ready_for_integration_evaluation_promotion`.

The contract always returns `live_authority_granted: false` and
`readiness_closure_allowed: false`.

## Why This Won

This makes integration evidence explicit without pretending that replay or
simulator evidence is hardware proof. It also gives tests a stable place to
verify child-contract blockers, safety-kernel trace binding, operator-report
binding and no-live-transport evidence.

## Rejected Alternatives

- Narrative-only integration tracking was rejected because it cannot fail
  closed when a child contract or integration trace is missing.
- Direct readiness promotion was rejected because offline integration evidence
  is not live WING proof and must not grant authority.

## Implementation Plan

1. Add the integration bundle contract module.
2. Add unit tests for incomplete child contracts, complete offline integration
   evidence, live or hardware claims, missing integration checks and forbidden
   live transport imports.
3. Bind the contract to `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`,
   `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` and the M013 rail.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

The main risk is treating a complete integration bundle as stronger evidence
than it is. The contract mitigates this by keeping live authority and readiness
closure false even when integration promotion is ready.

Rollback is to revert the M013 commit. No live console path imports or calls
this contract.
