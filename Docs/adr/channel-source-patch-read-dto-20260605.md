# ADR: Channel Source Patch Read DTO - 2026-06-05

## Context

The project goal requires the Automixer to read and analyze channel routing before it can safely control routing or build autonomous source-aware decisions. `build_channel_inventory` already publishes channel identity, static routing role, meters, basic controls, and cached processing coverage. It still uses fallback source labels such as `In 02` and does not normalize cached WING input source patch state.

WING source patch changes are destructive routing writes. They are not part of the current supervised live executor, which remains limited to approved Product Layer `fader` and `gain` writes.

## Options considered

1. Add cached source-patch DTO fields to channel inventory.
2. Query WING through `get_channel_input_routing`.
3. Leave source patch implicit until routing control is promoted.
4. Add routing proposals directly from cached patch state.

## Decision

Choose option 1. Channel inventory will expose cached main and ALT source patch state, read coverage, and cache freshness without querying or mutating hardware.

## Why this won

It improves console awareness while preserving the current safety boundary. Future routing proposals can inspect source group/channel evidence, but live routing remains blocked until physical readback, scoped diff, rollback proof, and operator review gates exist.

## Rejected alternatives

- Querying WING now was rejected because this Product Layer slice must not contact hardware.
- Leaving source patch implicit was rejected because fallback source labels are not enough for source-aware routing decisions.
- Creating routing proposals directly was rejected because this slice is read-only and route changes remain destructive.

## Implementation plan

1. Add a cached source-patch row builder for channel inventory.
2. Attach `source_patch` and source-patch coverage to every channel row.
3. Include source-patch addresses in channel cache freshness.
4. Update read-surface inventory and fixture evidence.
5. Add regression tests proving read-only behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py backend/read_surface_inventory.py backend/read_surface_fixture_evidence.py tests/test_operator_product_state.py tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_channel_inventory_publishes_cached_source_patch_readback_only tests/test_operator_product_state.py::test_channel_inventory_marks_missing_cached_read_sections_without_writes tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: callers may treat cached source patch as physical truth. The DTO keeps read-only safety flags, coverage metadata, cache freshness, and inventory blockers to prevent that claim.

Rollback is removing the `source_patch` DTO fields, inventory wording, fixture evidence, tests, and this ADR. No live write path is introduced.
