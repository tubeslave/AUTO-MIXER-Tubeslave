# Proposal: source-identity-calibration-persistence-plan-20260606

## Short Thesis

Add a read-only persistence plan for accepted source identity calibration records
so future memory/priority work has a validated manual candidate surface without
gaining write authority.

## What I Understood

The project needs long-term learning from accepted source identity calibration,
but writing models, memory, or downstream priority directly from acceptance
records would collapse review evidence into runtime authority. The safe next
step is a second validation boundary that turns accepted records into manual
persistence candidates only.

## Proposed Solution

- Add `source_identity_calibration_persistence_plan.py`.
- Revalidate the acceptance ledger envelope, safety flags, status, generated
  timestamp, summary counts, accepted/rejected candidate IDs, and summary safety
  claims.
- Revalidate accepted records before surfacing manual persistence candidates.
- Publish plan status in `main_pipeline_preview`.
- Register the plan as `shadow_only` source identity evidence in the processing
  registry and capability matrix.

## Likely Files To Touch

- `backend/source_identity_calibration_persistence_plan.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_persistence_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-persistence-plan-20260606.md`

## Alternatives Considered

- Let accepted ledger records write memory directly. Rejected because accepted
  review is not a persistence/write contract.
- Fold the persistence plan into the acceptance ledger. Rejected because a
  second fail-closed barrier protects future integrations from forged or stale
  ledger output.
- Skip persistence planning until a real memory writer exists. Rejected because
  the repo needs auditable candidate evidence before implementing writes.

## Risks

- Operators may assume manual persistence candidates have already changed trust
  state. Mitigate with explicit false flags for memory, model,
  downstream-priority, queue, and live mutations.
- A forged ledger summary could create false candidates. Mitigate by checking
  envelope identity, summary counts, summary safety flags, and accepted-record
  consistency.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue this is still too far from real memory writes. I would land it
first because real writes need this exact audited candidate boundary to avoid
turning acceptance into unreviewed mutation authority.
