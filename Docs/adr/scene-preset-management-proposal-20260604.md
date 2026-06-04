# ADR: Scene / Preset Management Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires automatic control of scenes and user presets. Existing
console clients and handlers contain direct scene/snapshot operations that can
overwrite broad mixer state. The current live executor is still limited to
supervised fader/gain writes, so scene and preset work must remain outside live
apply.

## 2. Options considered

- Leave scene/preset control blocked with no proposal contract.
- Add a proposal-only contract for scoped scene/preset recommendations.
- Promote direct scene/preset writes into Assisted or Supervised modes.

## 3. Decision

Add `scene_preset_management_proposal` as a transport-free proposal-only method.
Move `console_scene_preset_control` to `proposal_only` with an Assisted-mode
ceiling. Keep direct scene/snapshot/preset writes quarantined.

## 4. Why this won

This advances the required control surface without changing live authority. The
method records the evidence needed before any future promotion: current-state
readback, target preview, scoped diff, rollback snapshot, operator intent,
active-performance guard, active-signal guard, and scope limits.

## 5. Rejected alternatives

- Direct write promotion was rejected because scene recall and preset save can
  overwrite broad console state and lack hardware rollback proof.
- Keeping the feature only blocked was rejected because it leaves no structured
  path toward the project goal.
- Reusing routing proposals was rejected because scene/preset changes require
  broader state-diff evidence than route changes.

## 6. Implementation plan

1. Add a pure Python scene/preset analyzer with no mixer transport imports.
2. Emit only proposal payloads with `can_apply_live=False` and no
   `requested_change`.
3. Register the method as `proposal_only` in processing metadata.
4. Update the project-goal capability matrix to show Assisted proposal support.
5. Add tests for safe proposals, blockers, and no live widening.

## 7. Test plan

- Compile touched Python modules.
- Run focused scene/preset proposal tests.
- Run processing registry, proposal adapter, and project-goal matrix tests.
- Run `git diff --check`.

## 8. Risks and rollback

Risk: a UI may present scene/preset proposals as executable. Mitigation: payloads
include non-queueable flags, unsupported value-type blockers, and no requested
change. Rollback is to remove the method from registry/profile/matrix wiring;
the quarantined direct write paths remain unchanged.
