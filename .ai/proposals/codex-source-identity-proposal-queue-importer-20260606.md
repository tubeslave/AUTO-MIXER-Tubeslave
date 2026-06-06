# Proposal: source-identity-proposal-queue-importer-20260606

## Short Thesis

Add an operator-gated queue importer so source identity queue-ordering plans can
be turned into native operator proposal queue entries only through an explicit,
acknowledged import call.

## What I Understood

The system now has ranking evidence, an ordering preview, and a queue-ordering
plan. The next useful step is not live console control; it is a controlled
bridge into the backend-native queue, preserving the queue as an operator review
surface and keeping Product Layer preview non-mutating.

## Proposed Solution

- Add `source_identity_proposal_queue_importer.py`.
- Validate queue-ordering plan type, method, status, UTC timestamp, summary
  counts, plan records, duplicate IDs, and safety flags.
- Require source proposal payloads before import specs can be produced.
- Emit `queue_import_specs` in dry-run mode by default.
- Block `apply=true` unless `operator_ack=true`.
- Block acknowledged apply unless an explicit target queue with `create()` is
  provided.
- On acknowledged apply, call the target queue only; do not touch live mixer
  transport, models, runtime memory, or source identity trust stores.
- Wire Product Layer preview with `apply=false` and register the method as
  `shadow_only` decision evidence.

## Likely Files To Touch

- `backend/source_identity_proposal_queue_importer.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_proposal_queue_importer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-proposal-queue-importer-20260606.md`

## Alternatives Considered

- Import directly from queue-ordering plan in Product Layer. Rejected because
  previews must remain non-mutating.
- Mutate the server's queue implicitly. Rejected because queue mutation needs an
  explicit target and operator acknowledgement.
- Skip source proposal payloads and import skeletal queue rows. Rejected because
  real queue entries should preserve the original proposal payload when the
  operator chooses to import.

## Risks

- Imported queue entries may later be applyable through the existing supervised
  proposal path. Mitigate by preserving the existing approval/manual write gate;
  the importer itself never applies to console.
- Source payloads may drift from plan records. Mitigate by validating proposal
  IDs and duplicate IDs before specs or import results are emitted.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_queue_importer.py tests/test_source_identity_proposal_queue_ordering_plan.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_queue_importer.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that queue import should remain purely read-only until a UI
exists. I would keep the callable importer because it is still fail-closed,
requires explicit acknowledgement and target queue, and lets tests prove the
exact mutation boundary before UI work.
