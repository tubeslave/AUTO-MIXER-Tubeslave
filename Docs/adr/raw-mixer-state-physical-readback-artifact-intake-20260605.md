# Raw Mixer State Physical Readback Artifact Intake

## Context

Raw mixer-state physical proof is the next read-surface target. The repository
already has a schema contract and an operator capture plan, but no separate
release-visible intake status for captured artifacts.

## Options considered

1. Keep artifact validation only inside the raw mixer-state contract.
2. Add a separate read-only intake snapshot.
3. Build an executable hardware capture runner.

## Decision

Add `raw_mixer_state_physical_readback_artifact_intake` as a separate
shadow-only intake method. It validates separately captured JSON artifacts and
reports missing, valid candidate, or invalid candidate status.

## Why this won

The separate intake keeps the contract pure, makes artifact inventory visible,
and gives no-WING/Product Layer a fail-closed place to normalize unsafe source
claims. It advances the project toward physical readback proof without creating
a live query or write path.

## Rejected alternatives

Contract-only validation was less visible and harder to render in operator UI.
Executable capture was rejected because hardware capture must remain an
operator-controlled step until explicit physical proof and safety review exist.

## Implementation plan

- Add a transport-free backend intake builder.
- Register it as `shadow_only` in the processing registry and capability matrix.
- Validate and summarize it in no-WING release checks.
- Normalize it in Product Layer proof quality.
- Render it as a separate frontend Module Status row.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_raw_mixer_state_physical_readback_artifact_intake.py tests/test_raw_mixer_state_physical_readback_contract.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- Standard backend suite before merge.

## Risks and rollback

The main risk is treating artifact intake as fixture closure. Rollback is to
remove the intake method from no-WING/Product Layer/frontend surfaces; existing
contract and capture-plan behavior remain intact.
