# Proposal: scene-preset-cache-freshness-20260605

## 1. Short thesis

Add read-only cache freshness to the snapshots/scenes and user-presets Product
Layer DTOs, keeping scene/preset automation proposal-only and non-executable.

## 2. What I understood

Scene/preset management is broad console-state control. The project needs this
surface for the final AI Mixing Engineer goal, but current safety rules require
cached read contracts, freshness evidence, scoped diffs, rollback proof, and
hardware evidence before any apply path is considered.

## 3. Proposed solution

Reuse the existing `_cache_freshness_for_addresses()` helper in
`backend/operator_product_state.py` for scene/preset catalog and active-scene
addresses. Publish the resulting payload in `snapshots_scenes_snapshot` and
`user_presets_snapshot`.

Update read-surface inventory and fixture evidence to say that stale-cache
detection is now covered for cached DTO fixtures, while physical readback,
scoped diff, rollback diff, and parity gaps remain.

## 4. Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `.ai/briefs/scene-preset-cache-freshness-20260605.md`
- `Docs/adr/scene-preset-cache-freshness-20260605.md`

## 5. Alternatives considered

- Add scene/preset apply support. Rejected because scene recall/store can rewrite
  broad console state and is outside the approved supervised live executor.
- Require active OSC queries for freshness. Rejected because Product Layer
  snapshots must remain cached-only.
- Leave freshness as caller-provided proposal evidence. Rejected because it keeps
  the read DTO and proposal gate disconnected.

## 6. Risks

Fresh cached state may be mistaken for physical proof. Mitigation: preserve
`queries_console=False`, `live_write=False`, `full_autonomous_ready=False`, and
fixture evidence gaps for physical readback, scoped diff, and rollback.

## 7. Test plan

- Compile touched Python modules.
- Run focused Product Layer/read-surface tests.
- Run the standard repository pytest command if focused tests pass.

## 8. Where the other agent may disagree

Kimi may prefer implementing scoped scene/preset diff artifacts first. I would
still land freshness now because the existing scene/preset proposal gate already
depends on this evidence and the change does not widen live authority.
