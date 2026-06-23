# Codex Proposal: WING MVP Proof Plan Artifact Root Trace

## Short thesis

Make accepted-candidate `artifact_root` a required supervised proof-plan summary trace, parallel to the existing accepted-candidate prestart trace.

## What I understood

The WING read-only pipeline is still blocked on real hardware evidence. Offline gates should keep tightening the path from read-only candidate capture to supervised fader/gain proof planning without granting execution authority.

## Proposed solution

Add a report-only accepted-candidate artifact-root contract:

- Extract `artifact_root` from accepted `candidate_reviews`.
- Require each accepted `source_id` to have a non-empty root and live under that root.
- Expose actual and expected root diagnostics in `next_wing_proof_plan.summary`.
- Add a distinct blocker when the root contract is missing.
- Extend no-WING validation and summary rendering to recompute and compare those diagnostics.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- Project memory docs under `.ai/` and `Docs/`

## Alternatives considered

- Reuse the prestart blocker for root failures. Rejected because prestart and artifact-root traces are different contracts and should remain diagnosable.
- Trust the decision-review gate only. Rejected because the supervised proof plan is a downstream operator surface and must show the same root trace directly.

## Risks

- More summary fields increase fixture maintenance.
- Existing stale summaries will fail closed until regenerated.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_supervised_wing_proof_plan.py`
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py`
- Run project rails and lint checks before handoff.

## Where the other agent may disagree

Kimi may prefer moving the root helper into the shared contract module. I kept the helper local to the proof-plan/no-WING boundary because this slice only needs downstream summary validation and should not expand the shared preflight contract API unless reuse appears in another boundary.
