# Proposal: scene-preset-scoped-diff-evidence-20260605

## 1. Short thesis

Add structured, read-only scoped-diff evidence to scene/preset recommendations,
without creating any scene/preset executor.

## 2. What I understood

The final Automixer goal needs scene and user-preset control, but recalls and
stores can overwrite broad console state. The safe next step is richer evidence:
one scoped-diff payload should carry preview status, rollback status, affected
surface counts, and destructive-scope score.

## 3. Proposed solution

Add optional `scoped_diff_evidence` to `ScenePresetManagementAnalysisInput`.
Normalize it inside the analyzer and use it as a fallback/source for
`target_preview_available`, `scoped_diff_available`, rollback availability,
rollback ID, changed-parameter count, affected channel/bus/FX counts,
destructive-scope score, and confidence. If the diff payload is not read-only or
claims live write/query/mutation behavior, add blockers and keep the
recommendation `analysis_only`.

## 4. Likely files to touch

- `backend/scene_preset_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `tests/test_scene_preset_management_proposal.py`
- `tests/test_processing_registry.py`
- `.ai/briefs/scene-preset-scoped-diff-evidence-20260605.md`
- `Docs/adr/scene-preset-scoped-diff-evidence-20260605.md`

## 5. Alternatives considered

- Keep manual boolean fields only. Rejected because it is easy to forget which
  fields must agree.
- Generate diffs from live console state here. Rejected because this module is
  transport-free and must not query hardware.
- Treat diff evidence as apply authority. Rejected because scene/preset writes
  remain outside the supervised fader/gain executor.

## 6. Risks

The diff evidence contract may be mistaken for a runnable patch. Mitigation:
proposal payloads still include `can_import_to_queue=False`,
`can_apply_live=False`, and blockers for unsupported scene/preset value types.

## 7. Test plan

- Compile touched Python modules.
- Run focused scene/preset proposal tests.
- Run registry/contract tests.
- Run standard pytest before merge.

## 8. Where the other agent may disagree

Kimi may prefer building a separate Product Layer diff DTO first. I would add
the proposal evidence contract now because it gives that future DTO an explicit
target shape and fail-closed safety semantics.
