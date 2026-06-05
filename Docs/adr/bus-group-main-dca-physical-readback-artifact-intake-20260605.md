# ADR: Bus Group Main DCA Physical Readback Artifact Intake

## Context

`bus_group_main_dca_physical_readback_contract` defines the schema for future physical evidence covering bus controls, main controls, DCA controls, assignments, and metering. The project needs a release-visible place to inventory operator-captured artifacts without treating them as proof closure.

## Options Considered

- Do nothing and keep only the contract.
- Reuse a channel artifact intake.
- Add a dedicated bus/group/main/DCA artifact intake.

## Decision

Add a dedicated read-only artifact intake for `bus_group_main_dca_state` physical readback evidence.

## Why This Won

The dedicated intake can report bus, main, and DCA candidate counts while preserving the same fail-closed safety shape as the raw mixer-state and channel intakes.

## Rejected Alternatives

- Keeping only the contract leaves no structured place for operator-captured bus/main/DCA evidence.
- Reusing channel intake hides bus/DCA assignment and main-bus metering-specific validation.
- Letting the intake close fixture gaps is unsafe because candidate review and fixture promotion remain manual gates.

## Implementation Plan

- Add a transport-free backend intake builder.
- Add registry and capability matrix visibility.
- Add no-WING validator, summary output, and run-order hook.
- Add Product Layer normalization and frontend module-status exposure.
- Add focused tests for safe defaults, invalid authority claims, and UI contracts.

## Test Plan

- `tests/test_bus_group_main_dca_physical_readback_artifact_intake.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- no-WING quick release check and the standard pytest command before merge.

## Risks And Rollback

The main risk is that a valid artifact candidate is interpreted as live readiness. Roll back by removing the intake from registry, no-WING summary, Product Layer, and frontend exposure. The contract remains intact and no live authority is granted by this ADR.
