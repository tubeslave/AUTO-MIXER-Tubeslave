# Codex Proposal: WING Proof Plan Accepted Candidate Prestart Trace

## Short thesis

The supervised WING proof plan should only treat a read-only preflight review decision as accepted when the accepted source id is also present in `candidate_reviews` with the shared backend-port prestart contract.

## What I understood

Recent work made prestart fields visible in review reports and release summaries. The remaining weak point is the next proof plan boundary: it can see accepted source ids, but it did not prove that those ids came from prestart-valid candidate rows.

## Proposed solution

- Add helper logic in `backend/supervised_wing_proof_plan.py` to compute accepted candidate source ids that carry the exact shared prestart command and failure detail fields.
- Require those ids before marking `readonly_preflight_review_decision_accepted` true.
- Surface `readonly_preflight_review_decision_candidate_prestart_contract_valid` and source ids in `next_wing_proof_plan.summary`.
- Extend the no-WING validator and markdown summary to enforce and render the trace.

## Likely files to touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Trust `accepted_source_ids` alone. That leaves a gap where a stale or incomplete candidate row can still unlock physical proof steps.
- Re-validate the candidate artifact from disk inside the proof-plan builder. That would increase coupling and filesystem dependency in a module that currently consumes snapshots.

## Risks

- Existing external snapshots without `candidate_reviews` will now remain manual-review-ready instead of listing proof steps. That is intentional fail-closed behavior.
- The release summary gains two more fields. They are directly tied to the operator gate and should remain visible.

## Test plan

- Focused proof-plan and no-WING next-plan tests.
- Affected WING/no-WING pytest stack.
- Full standard pytest suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer validating candidate artifact hashes again at this boundary. I would keep this layer snapshot-based and rely on upstream artifact-hash gates, while requiring the accepted snapshot row to carry the prestart contract.
