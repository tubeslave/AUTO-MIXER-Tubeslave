# Codex Proposal: AutoPanner Apply Fail-Closed Boundary

## Short thesis

Keep AutoPanner as an intelligent panning decision layer, but make its direct apply paths fail closed so pan commands become blocked evidence instead of mixer mutations.

## What I understood

The project goal includes automatic panning and full console pan control. The repo already has `pan_spatial_proposal.py` as the transport-free assisted layer. Legacy `auto_panner.py` still had `apply_to_mixer()` calling `set_channel_pan`, and `auto_fader_v2/controller.py` had a simple pan-separation branch that also called `set_channel_pan`. Pan writes are not currently approved in the live executor.

## Proposed solution

- Add fail-closed policy fields to `AutoPanner`.
- Convert `AutoPanner.apply_to_mixer()` into a no-mutation method returning:
  - `applied=False`;
  - `calls=[]`;
  - `blocked_calls` with would-be `set_channel_pan` commands;
  - `mutates_mixer=False`.
- Convert `auto_fader_v2` pan separation into a no-mutation structured result with blocked pan commands.
- Update controller logging so blocked pan evidence is not described as applied mixer state.
- Add `set_channel_pan` to static write-token detection and write-surface inventory.
- Update registry, capability matrix, processing inventory, and tests.

## Likely files to touch

- `backend/auto_panner.py`
- `backend/auto_fader_v2/controller.py`
- `tests/test_auto_panner.py`
- `tests/test_write_surface_inventory_static_scan.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `Docs/adr/auto-panner-apply-fail-closed-20260605.md`

## Alternatives considered

- Delete `apply_to_mixer()`. Rejected because a compatibility method returning blocked evidence is safer for unknown callers.
- Keep controller pan separation live while blocking AutoPanner. Rejected because that would preserve a hidden pan write bypass.
- Promote pan writes now. Rejected because current-pan readback, mono compatibility, and rollback proof are not complete.

## Risks

- Legacy callers that expected an integer apply count now receive a structured blocked result.
- `set_channel_pan` command names remain in blocked evidence, so inventory/static-scan coverage must stay explicit.
- Controller pan separation may still calculate useful evidence but no longer mutates the mixer.

## Test plan

- Compile touched Python modules.
- Run AutoPanner, registry, capability matrix, and write-surface static scan tests.
- Run a runtime fake-mixer smoke test proving no `set_channel_pan` calls occur.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer renaming `apply_to_mixer()` immediately. I would keep it fail-closed for compatibility in this slice and consider a later rename after all callers are inventoried.
