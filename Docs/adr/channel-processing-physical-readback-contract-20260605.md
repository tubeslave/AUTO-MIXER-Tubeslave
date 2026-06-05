# ADR: channel-processing-physical-readback-contract-20260605

## Context

Full autonomous console awareness requires physical readback evidence for every
read surface. The repo already has cached DTO evidence for
`channel_eq_dynamics_gate`, but EQ, compressor, gate, de-esser, and unmasking
promotion targets still need physical proof that channel processing state
matches the actual console.

## Options considered

1. Add a dedicated channel processing physical readback contract.
2. Reuse the channel basic-control readback contract.
3. Wait for a hardware proof runner before defining the artifact contract.

## Decision

Use option 1. Add `channel_processing_physical_readback_contract` as a
transport-free, shadow-only contract. The contract validates future operator-run
artifacts with HPF, EQ bands, compressor, gate, and coverage fields, but does
not query hardware or grant live authority.

## Why this won

The channel processing surface is a promotion dependency for several write
families, and its proof shape is richer than channel identity/basic-control
rows. A dedicated schema lets no-WING release checks, Product Layer summaries,
and cockpit module status agree on what evidence is still missing before any
promotion review.

## Rejected alternatives

The channel basic-control contract cannot express EQ-band, dynamics, and
coverage requirements without weakening both schemas. Waiting for hardware proof
would leave future artifacts underspecified and easier to misinterpret as
authority.

## Implementation plan

1. Add the backend contract module and focused validation tests.
2. Register it as `shadow_only` and add it to the console-read capability row.
3. Add fixture/remediation declarations while preserving the physical gap.
4. Validate the contract in no-WING release checks and summaries.
5. Expose normalized Product Layer/frontend status rows.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/channel_processing_physical_readback_contract.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_channel_processing_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend test -- --watch=false`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: contract readiness may be mistaken for hardware proof. Mitigation:
summary and UI fields keep physical fixture gaps and live authority false.

Rollback is removing the contract module, release/Product Layer/frontend wiring,
tests, and this ADR. No live transport path is added.
