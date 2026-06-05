# Codex Proposal: AutoFOH Safety Direct Apply Fail Closed - 2026-06-05

## Short thesis

Keep AutoFOH bounded typed correction decisions, but make every direct apply surface outside channel fader/gain fail closed as blocked evidence.

## What I understood

AutoFOH Safety is valuable because it normalizes actions, applies bounds and enforces runtime policy. The unsafe part is that its translator can still directly call broad mixer methods outside the current Product Layer proof boundary.

## Proposed solution

- Add a proposal-only write policy to `AutoFOHSafetyController`.
- After runtime policy and bounds, intercept non-channel-fader/gain actions before any mixer translator.
- Return blocked evidence containing the would-be command and bounded values.
- Keep channel fader/gain direct translator behavior unchanged for this slice.
- Update focused tests, inventory, capability matrix, processing inventory and ADR.

## Likely files to touch

- `backend/autofoh_safety.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_autofoh_safety.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/autofoh-safety-direct-apply-fail-closed-20260605.md`

## Alternatives considered

- Leave AutoFOH Safety as the broad executor because it has bounds. Rejected because bounds are not proof of readback, rollback or hardware safety.
- Block all AutoFOH actions including channel fader/gain. Deferred because that is a separate Product Layer migration and would touch more runtime bridge behavior.
- Delete typed non-fader actions. Rejected because their bounded values are useful for proposal evidence and future promotion.

## Risks

- Tests or callers that expected AutoFOH EQ/compressor/gate/pan/phase/send/bus writes to mutate fake mixers now receive blocked decisions.
- `allowed=False` now means runtime may have allowed the action family, but the current write policy blocked direct apply.

## Test plan

- Update AutoFOH Safety tests to assert blocked evidence and no fake mixer mutation.
- Add a static source test proving direct non-fader/gain mixer calls are absent from `autofoh_safety.py`.
- Run focused AutoFOH, registry, capability and affected soundcheck tests.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer moving channel fader/gain through Product Layer in the same patch. That is the right long-term direction, but this slice removes broad hidden writes first while keeping the blast radius controlled.
