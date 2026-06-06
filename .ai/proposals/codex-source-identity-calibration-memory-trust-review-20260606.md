# Proposal: source-identity-calibration-memory-trust-review-20260606

## Short Thesis

Add a read-only trust-review layer that consumes indexed source identity memory
and produces operator-review candidates only with fresh replay context, without
granting downstream priority authority.

## What I Understood

The Automixer needs long-term source identity learning to eventually influence
future decisions, but each promotion boundary must remain explicit. Indexed
memory proves that an operator-acknowledged memory record exists; it does not
prove the current source context is fresh or that downstream proposal priority
may change.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_review.py`.
- Revalidate memory index envelope, status, safety flags, summary counts,
  identity lists, and record false-authority flags.
- Require fresh source identity replay context before surfacing review
  candidates.
- Emit review candidates with operator-review-required state and false runtime,
  model, queue, downstream-priority, and live-authority flags.
- Wire review status into `main_pipeline_preview`.
- Register the review layer as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_review.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_review.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-trust-review-20260606.md`

## Alternatives Considered

- Let indexed memory directly feed `source_identity_downstream_gate`. Rejected
  because memory existence is not enough to trust the current source context.
- Keep indexed memory visible only as raw evidence. Rejected because future
  promotion work needs a stable review-candidate contract.
- Make trust review queueable. Rejected because this is not a live action and
  should not enter the supervised fader/gain executor path.

## Risks

- Review candidates may be mistaken for downstream trust. Mitigate by forcing
  `downstream_priority_change_allowed_by_review=false` and keeping candidates
  out of the decision queue.
- Stale replay context could make old memory look current. Mitigate by requiring
  fresh source identity replay context before candidates are surfaced.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that fresh indexed memory should immediately change downstream
proposal priority. I would keep that as a separate promotion contract because
review candidates still need replay confirmation, operator acceptance, and
rollback semantics before they influence decision behavior.
