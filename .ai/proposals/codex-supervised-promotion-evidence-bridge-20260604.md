# Codex Proposal: Supervised Promotion Evidence Bridge

## 1. Short thesis

Connect proof artifact reports to safety coverage rows with a read-only bridge
that can justify manual review without changing live authority.

## 2. What I understood

The project needs to move toward supervised and eventually autonomous control,
but only through proof. We can now evaluate proof artifacts and write reports.
The next gap is deciding whether those reports are strong enough to consider a
manual safety-row update for fader or gain.

## 3. Proposed solution

Create `backend/supervised_promotion_evidence.py` with rows for fader and gain.
Each row reads the proof report and current safety coverage row, then reports:

- physical candidate count;
- dry-run-ready count;
- blocked artifact count;
- current safety status and live-allowed state;
- whether manual safety-row review is justified;
- blockers and next required evidence.

The bridge always sets automatic safety update, live widening, and autonomous
ready to false.

## 4. Likely files to touch

- `backend/supervised_promotion_evidence.py`
- `tests/test_supervised_promotion_evidence.py`
- `tools/evaluate_supervised_proof_artifacts.py`
- `tests/test_supervised_proof_artifact_report.py`
- `backend/processing_registry.py`
- `tests/test_processing_registry.py`
- `Docs/adr/supervised-promotion-evidence-bridge-20260604.md`

## 5. Alternatives considered

- Directly update `safety_coverage_matrix.py` from proof reports: rejected
  because proof evidence must go through manual review.
- Keep report interpretation manual: rejected because that makes readiness
  claims hard to test.
- Include broad EQ/dynamics/FX rows now: rejected because only fader/gain share
  the supervised Product Layer executor.

## 6. Risks

- The bridge could be mistaken for promotion authority. It explicitly reports
  review candidacy only and denies automatic update.
- Existing artifacts may contain dry-run-only evidence. The bridge keeps that
  as `dry_run_only` or `evidence_blocked`.
- A blocked artifact for the same value type must stop review candidacy until
  the failure is understood.

## 7. Test plan

- Verify the bridge is transport-free.
- Verify dry-run-only gain cannot justify safety-row review.
- Verify physical gain proof permits manual review but not live widening.
- Verify blocked and invalid proof evidence blocks review.
- Verify registry keeps the bridge outside live executors.

## 8. Where the other agent may disagree

Kimi may prefer to immediately run real WING proof. That is the next hardware
step, but this bridge prevents any future proof result from being interpreted
as live authority without tested manual review semantics.
