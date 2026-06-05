# Codex Proposal: AutoReverb Apply Fail-Closed Boundary

## Short thesis

Keep AutoReverb as an intelligent reverb decision layer, but make its legacy apply method fail closed so reverb send/parameter commands become blocked evidence instead of mixer mutations.

## What I understood

AutoReverb computes musically useful decisions: reverb type, decay, level, predelay, brightness, density, HPF, and LPF per channel. The unsafe part is not the analysis; it is `apply_to_mixer()`, which directly calls `set_reverb_send` and `set_reverb_params`. That path can be reached from `auto_fader_v2` when auto reverb is enabled, but direct FX/reverb writes are outside the current live-approved executor.

## Proposed solution

- Add `live_write_allowed=False` and `write_policy="proposal_only_quarantined"` to `AutoReverb`.
- Convert `apply_to_mixer()` into a no-mutation method returning:
  - `applied=False`;
  - `calls=[]`;
  - `blocked_calls` with the would-be reverb send and parameter commands;
  - `mutates_mixer=False`.
- Add a helper to build blocked reverb command evidence from `ReverbDecision` values.
- Update `auto_fader_v2` logging so blocked evidence is not described as applied mixer state.
- Update registry, write-surface inventory, capability matrix, processing inventory, and tests.

## Likely files to touch

- `backend/auto_reverb.py`
- `backend/auto_fader_v2/controller.py`
- `tests/test_auto_reverb.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `Docs/adr/auto-reverb-apply-fail-closed-20260605.md`

## Alternatives considered

- Delete `apply_to_mixer()`. Rejected because a compatibility method returning blocked evidence is safer for existing callers.
- Keep direct apply only behind the auto-fader controller. Rejected because that would preserve a hidden live write path.
- Promote reverb writes to supervised auto mode now. Rejected because reverb send/return readback, feedback checks, and rollback proof are not complete.

## Risks

- Legacy callers that expected an integer apply count now receive a structured blocked result.
- The `apply_to_mixer()` name remains historical and can mislead readers unless tests and inventory stay explicit.
- Blocked evidence still contains write command names, so static inventory coverage must remain in place.

## Test plan

- Compile touched Python modules.
- Run AutoReverb, registry, capability matrix, and write-surface scan tests.
- Run the full backend suite.
- Run a runtime fake-mixer smoke test proving no `set_reverb_*` calls occur.

## Where the other agent may disagree

Kimi may prefer deleting or renaming `apply_to_mixer()`. I would keep it fail-closed in this slice because it closes the safety hole immediately and gives legacy callers a reviewable result.
