# ADR: Auto Gate CAIG Fail-Closed Write Policy

## Context

CAIG computes gate decisions from audio features, adaptive thresholds, group context, and drum rules. The analysis path is useful for assisted mixing and future gate proposals. However, historical direct gate-send code is outside the current live safety boundary. Current live apply remains limited to supervised Product Layer fader/gain writes.

Gate writes need stricter proof than recommendation output because a bad gate decision can cut sources, open bleed, or destabilize a live mix. The repository already classifies gate/HPF/send direct apply as quarantined.

## Options considered

1. Keep CAIG analysis but make direct sends fail closed.
2. Delete CAIG direct send compatibility immediately.
3. Add an opt-in flag for live gate writes.
4. Quarantine the entire CAIG method.

## Decision

Choose option 1. CAIG remains a proposal-only analysis source. The legacy `_send_osc()` method is kept for compatibility but becomes a quarantine sink that records blocked attempts and performs no mixer mutation.

## Why this won

This preserves useful gate-decision analysis while removing the unsafe live-write behavior. It also aligns runtime behavior with the write-surface inventory: gate/HPF/send direct apply remains quarantined, and the only approved live executor remains supervised fader/gain.

## Rejected alternatives

- Deleting `_send_osc()` now could break unknown legacy callers before the caller inventory is complete.
- A live-write flag would create an unproven bypass around false-open/false-close replay, physical readback, and rollback proof.
- Quarantining all CAIG logic would throw away useful analysis that can still support proposal-only review.

## Implementation plan

- Reset CAIG write telemetry at the start of each `process()` call.
- Make `_send_osc()` return `0` without calling mixer-client mutation methods.
- Expose fail-closed status fields through `get_status()`.
- Record the boundary in `processing_registry`, `write_surface_inventory`, and the project goal capability matrix.
- Add fake-mixer tests proving both `process()` and `_send_osc()` do not call `set_channel_gate`.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/auto_gate_caig.py backend/processing_registry.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_gate.py tests/test_processing_registry.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_gate.py tests/test_processing_registry.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- Runtime fake-mixer smoke test for no mutation.

## Risks and rollback

Risk: a legacy caller that expected CAIG to apply gates will now only receive dry-run decisions. This is intentional because gate writes are not approved. Rollback should not re-enable direct writes; if behavior must change, introduce a supervised proposal/apply contract with fake-mixer tests, physical readback evidence, rollback values, and an ADR.
