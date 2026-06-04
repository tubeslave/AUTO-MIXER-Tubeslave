# Codex Proposal: Autonomy Evidence Overlay

Task id: autonomy-evidence-overlay-20260604
Date: 2026-06-04

## 1. Short thesis

Fresh proof evidence should be visible at the autonomy-promotion target level,
but it must not mutate the static safety matrix or widen live authority.

## 2. What I understood

The static readiness snapshot and proof report answer related but separate
questions. Static readiness says which action families are policy/proof
blocked. The proof report says which current artifacts are dry-run-ready,
physical candidates, or blocked. The missing piece is a tested overlay that
connects current artifacts to promotion targets.

## 3. Proposed solution

Add `autonomy_evidence_overlay` with rows for the current proof families:
fader and gain. Each row maps to the corresponding promotion target and reports
physical candidate ids, dry-run ids, blocked ids, remediation action ids,
status, blockers, and next actions.

## 4. Likely files to touch

- `backend/autonomy_evidence_overlay.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `backend/processing_registry.py`
- focused tests and ADR/brief artifacts

## 5. Alternatives considered

- Fold dynamic proof evidence into the static readiness snapshot.
- Leave proof evidence separate from readiness.
- Auto-update safety rows from physical candidates.

## 6. Risks

Operators may read a manual review candidate as automatic permission. The
overlay must keep `live_write=false`, `live_widening_allowed=false`,
`automatic_safety_update_allowed=false`, and `autonomous_ready=false`.

## 7. Test plan

- Unit-test physical fader candidate and legacy gain remediation mapping.
- Unit-test dry-run-only and missing-evidence states.
- Assert proof report embedding and registry shadow-only status.
- Run focused, broad, smoke, and full pytest validation.

## 8. Where the other agent may disagree

Kimi may prefer integrating dynamic proof evidence directly into
`autonomy_promotion_readiness`. Keeping it separate is safer because current
artifacts should inform manual review without rewriting static policy.
