# ADR: bus-group-main-dca-physical-readback-contract-20260605

## Context

Full autonomous console awareness requires physical readback evidence for every
read surface. The repo already has cached DTO evidence for
`bus_group_main_dca_state`, but grouped balance, DCA, main-bus optimization,
and performance-adaptation promotion targets still need physical proof that
bus, main, and DCA state match the actual console.

## Options considered

1. Add a dedicated bus/group/main/DCA physical readback contract.
2. Reuse channel control or raw mixer-state readback contracts.
3. Wait for a hardware proof runner before defining the artifact contract.

## Decision

Use option 1. Add `bus_group_main_dca_physical_readback_contract` as a
transport-free, shadow-only contract. The contract validates future operator-run
artifacts with bus controls, main controls, DCA controls, assignment, metering,
and coverage fields, but does not query hardware or grant live authority.

## Why this won

The bus/group/main/DCA surface is a promotion dependency for several mix-balance
write families, and its proof shape is richer than channel rows or raw
address/value rows. A dedicated schema lets no-WING release checks, Product
Layer summaries, and cockpit module status agree on what evidence is still
missing before any promotion review.

## Rejected alternatives

The channel contracts cannot express DCA membership, bus-to-main assignment,
and bus/main metering requirements without weakening both schemas. Waiting for
hardware proof would leave future artifacts underspecified and easier to
misinterpret as authority.

## Implementation plan

1. Add the backend contract module and focused validation tests.
2. Register it as `shadow_only` and add it to the console-read capability row.
3. Add fixture/remediation declarations while preserving the physical gap.
4. Validate the contract in no-WING release checks and summaries.
5. Expose normalized Product Layer/frontend status rows.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/bus_group_main_dca_physical_readback_contract.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_bus_group_main_dca_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
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
