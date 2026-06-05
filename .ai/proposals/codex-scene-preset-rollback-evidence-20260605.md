# Proposal: scene-preset-rollback-evidence-20260605

## 1. Short thesis

Add structured rollback evidence for scene/preset proposals while keeping all
scene and preset operations non-executable.

## 2. What I understood

The final Automixer goal needs scene and preset control, but scene recall/store
can rewrite broad console state. Rollback has to become a first-class evidence
object before any future assisted or supervised promotion is credible.

## 3. Proposed solution

Add optional `rollback_evidence` to `ScenePresetManagementAnalysisInput`. Normalize
it into availability, rollback snapshot ID, confidence, target-scope coverage,
and safety flags. The analyzer should use it as a source for rollback availability
and confidence, and add blockers if the evidence is not read-only, claims live
write/query behavior, mutates console state, or explicitly does not cover the
target scope.

## 4. Likely files to touch

- `backend/scene_preset_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_scene_preset_management_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/scene-preset-rollback-evidence-20260605.md`
- `Docs/adr/scene-preset-rollback-evidence-20260605.md`

## 5. Alternatives considered

- Keep rollback as only a boolean and string ID. Rejected because it cannot
  distinguish read-only proof from executable rollback behavior.
- Execute rollback validation in this module. Rejected because this module is
  transport-free and must not contact hardware.
- Treat scoped-diff rollback fields as enough. Rejected because rollback coverage
  deserves explicit safety flags and target matching.

## 6. Risks

Rollback evidence may be mistaken for rollback authority. Mitigation: proposal
payloads remain non-queueable and non-applicable, and registry notes keep
`no_live_writes`.

## 7. Test plan

- Compile touched modules.
- Focused scene/preset proposal tests.
- Registry/capability tests.
- Standard pytest before merge.

## 8. Where the other agent may disagree

Kimi may prefer waiting for physical rollback fixtures. I would still land the
contract now because it defines the fail-closed shape those fixtures must satisfy.
