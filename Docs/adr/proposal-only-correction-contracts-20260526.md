# Proposal-Only Correction Contracts

## Context

The project contains multiple correction domains: channel fader/gain, EQ,
compressor, gate, pan, phase/delay, FX/sends, routing, output paths, and
offline render/evaluation systems. Only the supervised channel `fader` and
`gain` path has the current safe live executor shape for WING.

Without an explicit Product Layer contract, UI or backend code can accidentally
make a non-proven domain look applyable just because it has a `requested_change`
shape.

## Options considered

1. Promote EQ, compressor, gate, pan, FX, and routing into live apply now.
2. Hide all non-fader/gain methods from the cockpit.
3. Expose them as proposal-only contracts with required evidence, while keeping
   live apply blocked until separate proof.

## Decision

Use option 3.

`main_live_assist` now exposes proposal-only correction contracts for non-fader
and non-gain domains. These contracts show what a future safe proposal must
contain: target, old value, proposed value, safety evidence, required approval,
readback requirement, and rollback value.

The operator proposal queue treats only `fader` and `gain` as applyable value
types. Unsupported domains can still be represented for review, but the queue
marks them as not applyable and records explicit blockers.

## Why this won

This gives the frontend and Product Layer a truthful view of existing project
capabilities without widening the live write surface. It also creates a concrete
promotion path for later hardware validation.

## Rejected alternatives

Direct live promotion was rejected because EQ, compressor, gate, pan, FX,
routing, and phase/delay do not yet have the same WING proof as fader/gain.

Hiding the methods was rejected because the operator cockpit needs to show what
exists, what is blocked, and what evidence is missing.

## Implementation plan

- Add proposal-only contract builders to the processing proposal adapter layer.
- Include those contracts in `get_main_pipeline_preview`.
- Keep them separate from queueable fader/gain proposals.
- Harden `OperatorProposalQueue` so unsupported value types are not presented as
  applyable.
- Add tests for read-only visibility and non-applyable queue behavior.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_operator_product_state.py -q`
- `git diff --check`

## Risks and rollback

Risk: UI may need to distinguish queueable proposals from proposal-only
contracts. Mitigation: contracts use a separate `proposal_only_contracts` field
and `queue_action=not_queueable_until_promoted`.

Rollback: remove the proposal-only contract field, adapter, tests, and this ADR.
No live write path is widened by this decision.
