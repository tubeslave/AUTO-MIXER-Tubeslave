# Proposal: source-identity-calibration-memory-trust-acceptance-ledger-20260606

## Short Thesis

Add a read-only ledger for operator decisions on memory-trust review candidates,
keeping accepted memory trust as evidence only until a separate promotion
contract exists.

## What I Understood

The Automixer needs durable source identity memory to eventually improve future
decisions. But an operator accepting a memory-trust candidate still must not
change runtime trust or downstream priority by itself. Acceptance is a record in
the chain, not the promotion gate.

## Proposed Solution

- Add `source_identity_calibration_memory_trust_acceptance_ledger.py`.
- Revalidate memory trust-review envelope, status, safety flags, summary counts,
  and candidate false-authority flags.
- Validate operator acceptance records against candidate IDs, source identity
  fields, decision values, UTC timestamps, and unsafe authority claims.
- Surface accepted/rejected records as read-only ledger evidence.
- Wire ledger counts into `main_pipeline_preview`.
- Register the ledger as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_acceptance_ledger.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-trust-acceptance-ledger-20260606.md`

## Alternatives Considered

- Let accepted memory-trust records directly change downstream priority.
  Rejected because promotion still needs replay, rollback, and explicit
  trust-gate semantics.
- Keep review candidates without acceptance records. Rejected because the
  system needs an auditable operator decision point.
- Make accepted records queueable. Rejected because memory trust acceptance is
  not a mixer action.

## Risks

- Accepted records may be mistaken for promotion authority. Mitigate by forcing
  all promotion, queue, model, runtime-memory, downstream-priority, and live
  flags false.
- Forged acceptance records could claim live or priority authority. Mitigate by
  rejecting unsafe flags and mismatched candidate fields.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_acceptance_ledger.py tests/test_source_identity_calibration_memory_trust_review.py tests/test_source_identity_calibration_memory_index.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_acceptance_ledger.py backend/source_identity_calibration_memory_trust_review.py backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that operator acceptance is enough to update downstream trust.
I would keep acceptance separate from promotion because priority changes still
need a contract that proves freshness, rollback, and family-specific behavior.
