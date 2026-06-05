# ADR: AutoReverb Apply Fail-Closed Boundary

## Context

`backend/auto_reverb.py` computes per-channel reverb decisions using instrument profiles, spectral centroid, spectral flux, dry-side margins, predelay, and reverb filtering rules. The analysis is useful for the target AI mixing engineer. The legacy `apply_to_mixer()` method, however, directly called mixer-client `set_reverb_send` and `set_reverb_params` methods.

Direct reverb writes are part of the quarantined FX/send surface. They can change intelligibility, feedback risk, manual engineer intent, and wet/dry balance. Current live execution remains limited to supervised fader/gain, so reverb decisions must stay reviewable evidence until readback, feedback checks, and rollback proof exist.

## Options considered

1. Keep AutoReverb analysis but make `apply_to_mixer()` fail closed.
2. Delete `apply_to_mixer()` immediately.
3. Keep direct apply through `auto_fader_v2` only.
4. Promote reverb writes into supervised auto mode now.

## Decision

Choose option 1. AutoReverb remains an analysis/proposal layer. `apply_to_mixer()` stays for compatibility but returns blocked apply evidence and performs no mixer-client mutation.

## Why this won

This preserves useful reverb decisions while removing a hidden direct-write path reachable from the auto-fader controller. It aligns AutoReverb with the proposal-only FX/send architecture and keeps the live executor limited to proven supervised fader/gain.

## Rejected alternatives

- Deleting the method could break legacy callers without giving them safe blocked evidence.
- Keeping controller-only direct apply would leave an unsafe bypass around the write-surface inventory.
- Promoting reverb writes now would skip required send/return readback, feedback, rollback, and physical proof.

## Implementation plan

- Add fail-closed policy fields to AutoReverb.
- Convert `apply_to_mixer()` into a no-mutation method returning `blocked_calls`.
- Build blocked `set_reverb_send` and `set_reverb_params` command records from decisions.
- Update `auto_fader_v2` logging for blocked reverb evidence.
- Update registry, write-surface inventory, capability matrix, processing inventory, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/auto_reverb.py backend/auto_fader_v2/controller.py backend/processing_registry.py backend/write_surface_inventory.py backend/project_goal_capability_matrix.py tests/test_auto_reverb.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_auto_reverb.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- Runtime fake-mixer smoke test for no mutation.

## Risks and rollback

Risk: a legacy caller that expected an integer apply count now receives a structured blocked result. This is intentional because direct reverb writes are not approved. Rollback must not restore direct `set_reverb_*` calls; a live path requires a supervised proposal/apply contract with physical readback, feedback checks, bounded wet/dry/send limits, rollback values, and a separate ADR.
