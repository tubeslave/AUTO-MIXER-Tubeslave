# Proposal: WING MVP Accepted Candidate Expected Artifact Path Contract

## 1. Short Thesis

Make the shared accepted-candidate prestart contract compare expected artifact
paths with actual summary/event paths instead of trusting match flags alone.

## 2. What I Understood

The current WING MVP rail must keep supervised proof blocked until the accepted
read-only candidate is bound to the exact real artifacts and a valid no-WING
prestart trace.

## 3. Proposed Solution

- Extend `_candidate_expected_artifact_binding_violations()` to compare
  `expected_summary_artifact` with `summary_path` and
  `expected_events_artifact` with `events_path`.
- Keep existing boolean match-flag checks.
- Add regression coverage in the review-decision ledger helper and no-WING
  proof-plan validator.

## 4. Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`

## 5. Alternatives Considered

Only validating this in the review template is insufficient because no-WING
validation and proof-plan summaries use the shared contract helper directly.

## 6. Risks

Older synthetic fixtures may omit `summary_path` or `events_path`; the change
therefore compares against present actual paths while preserving the existing
flags.

## 7. Test Plan

Run focused review-decision, supervised proof-plan, and no-WING validator tests,
then run the broader WING-focused suite and the standard pytest command.

## 8. Where The Other Agent May Disagree

Kimi may ask for a stricter schema requiring both actual paths unconditionally.
That is reasonable for real evidence rows, but this scoped change avoids
breaking older report fixtures while still closing the live proof-plan path.
