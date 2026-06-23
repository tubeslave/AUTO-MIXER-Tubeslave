# WING MVP Proof Plan Artifact Root Trace

## Summary

The supervised WING proof plan now carries the accepted read-only preflight candidate `artifact_root` as a first-class report-only trace. A missing or mismatched root keeps the plan at `manual_review_ready` and prevents physical proof steps from being listed.

## Changed

- `backend/supervised_wing_proof_plan.py` derives accepted-candidate artifact roots from the review-decision ledger.
- `tools/no_wing_release_check.py` recomputes expected roots from the ledger and rejects stale `next_wing_proof_plan` summaries.
- `backend/wing_readonly_preflight_contract.py` defines `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`.
- Focused tests cover the happy path, missing accepted-candidate root, and stale no-WING summaries.

## Safety Boundary

This is an offline/report-only gate. It does not contact WING, open mixer transport, grant live authority, or authorize fader/gain execution. It only blocks supervised proof planning unless the accepted read-only candidate root trace is visible and internally consistent.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_supervised_wing_proof_plan.py` passed with 16 tests.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py` passed with 274 tests and 8 skips.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and operator-reviewed accepted candidate before any supervised fader/gain proof run.
