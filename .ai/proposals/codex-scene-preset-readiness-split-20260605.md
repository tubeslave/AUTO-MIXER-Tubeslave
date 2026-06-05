# Proposal: scene-preset-readiness-split-20260605

## 1. Short thesis

Separate scene/preset promotion readiness from routing readiness so future
automation can prove them independently while both remain fail-closed today.

## 2. What I understood

The project goal includes full console control, including routing, scenes, and
user presets. The current readiness row combines those surfaces into one broad
target. That hides whether scene/preset-specific work such as Product Layer
snapshot freshness, scoped diffs, and read-only rollback evidence is progressing.

## 3. Proposed solution

Replace the combined `routing_scene_preset_supervised_promotion` target with two
targets:

- `routing_supervised_promotion`: routing-only capability plus
  `routing_and_sends` read surface.
- `scene_preset_supervised_promotion`: scene/preset-only capability plus
  `snapshots_scenes` and `user_presets` read surfaces.

Both can continue to use the existing blocked `routing_snapshots_scenes` safety
coverage row and quarantined `routing_snapshot_scene_writes` write surface until
a later ADR creates separate proof rows.

## 4. Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `.ai/briefs/scene-preset-readiness-split-20260605.md`
- `Docs/adr/scene-preset-readiness-split-20260605.md`

## 5. Alternatives considered

- Keep one combined row. Rejected because it obscures independent readiness and
  makes scene/preset rollback evidence hard to inspect.
- Add a new safety row now. Rejected because no separate physical proof exists.
- Promote scene/preset proposals to queueable. Rejected because broad console
  state writes are still blocked.

## 6. Risks

The split might be misread as promotion. Mitigation: preserve the blocked safety
row, quarantined write surface, `supervised_live_allowed=False`, and
`autonomous_ready=False` assertions.

## 7. Test plan

- Focused autonomy promotion readiness tests.
- Processing registry snapshot test that includes autonomy readiness.
- Standard pytest command.

## 8. Where the other agent may disagree

Kimi may prefer creating separate safety coverage rows immediately. I would defer
that until physical/fake proof exists; this slice is only visibility and
fail-closed planning.
