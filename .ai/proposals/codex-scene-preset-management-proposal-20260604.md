# Codex Proposal: Scene / Preset Management Proposal Layer

## 1. Short thesis

Add a scene/preset proposal-only method that turns scoped scene and user-preset
intent into operator-review evidence while keeping all direct scene, snapshot,
and preset writes outside the live executor.

## 2. What I understood

The long-term Automixer goal requires scene and preset control, but scene recall
or preset save can overwrite broad console state. The current proven live write
boundary is still supervised fader/gain only. The correct next step is not live
scene automation; it is a safe proposal contract with readback, diff, rollback,
scope, and intent evidence.

## 3. Proposed solution

Create `backend/scene_preset_management_proposal.py` with a transport-free
dataclass API:

- normalize scene/preset artifact type, action, and scope;
- evaluate current-state readback, target preview, scoped diff, rollback
  snapshot, operator intent, safe window, active-signal guard, and scope limits;
- emit `proposal_only` recommendations only when evidence is complete;
- emit Product Layer style proposals with `kind="scene_preset"`,
  `can_import_to_queue=False`, `can_apply_live=False`, and no
  `requested_change`.

Register the method in the processing registry and capability matrix as
`proposal_only` / `assisted`, while leaving the existing direct
`routing_snapshot_reset_writes` method quarantined.

## 4. Likely files to touch

- `backend/scene_preset_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_scene_preset_management_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/scene-preset-management-proposal-20260604.md`

## 5. Alternatives considered

- Promote direct scene recall/save into Assisted mode: rejected because broad
  console state can be overwritten and hardware rollback proof is missing.
- Keep the row blocked with no contract: rejected because it does not move the
  project toward the required scene/preset control surface.
- Route scene/preset through the existing routing proposal: rejected because
  scenes and presets require full-state diff and rollback evidence, not only
  route-diff evidence.

## 6. Risks

- Operators may misread proposal-only scene/preset output as executable.
- Scoped diffs may be incomplete until read coverage improves.
- Save/overwrite proposals need especially strict intent and rollback evidence.

## 7. Test plan

- Assert the module is free of mixer clients and direct scene/snapshot methods.
- Assert safe scoped recall produces a proposal-only payload with no live apply.
- Assert active performance, broad scope, active signal, missing preview, and
  missing rollback block proposals.
- Assert registry, proposal contract, and project-goal matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue to keep scene/preset control fully blocked until read coverage is
complete. The counterpoint is that proposal-only metadata is the safer path to
define required evidence without granting write authority.
