# Proposal: bus-mute-evidence-requirements-20260605

## 1. Short thesis

Make bus/group/matrix and mute proof gates machine-readable in promotion
readiness without widening live control.

## 2. What I understood

The previous safety slice replaced generic missing safety rows with explicit
fail-closed contracts. The next gap is that the promotion rollup still cannot
name the exact evidence artifacts needed for bus/group/matrix and mute
promotion the way it already does for routing and scene/preset targets.

## 3. Proposed solution

Add `evidence_requirement_ids` to the two promotion targets:

- bus/group/matrix: assignment freshness, fader readback, matrix send readback,
  true-peak/headroom evidence, physical fixture, and hardware rollback proof.
- mute: current mute freshness, owner/intent evidence, signal-return evidence,
  manual authority review, physical fixture, and hardware emergency/rollback
  proof.

## 4. Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `Docs/adr/bus-mute-evidence-requirements-20260605.md`

## 5. Alternatives considered

- Leave requirements only in prose. Rejected because proof runners and cockpit
  gap summaries need stable IDs.
- Add frontend-only labels. Rejected because the backend readiness contract
  should own promotion evidence semantics.

## 6. Risks

Operators may overread evidence IDs as apply readiness. Tests should preserve
`supervised_live_allowed=false` and `autonomous_ready=false` for both rows.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may ask for a reusable evidence-ID registry. That is reasonable later,
but the current readiness target dataclass already owns these IDs for routing
and scene/preset, so this slice follows the local pattern.
