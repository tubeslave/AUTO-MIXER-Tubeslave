# ADR: source-identity-calibration-memory-index-20260606

## Context

Source identity calibration has an operator-reviewed path from replay proposals
to acceptance records, manual persistence candidates, append-only memory drafts,
and an operator-acknowledged writer. The next gap is read visibility: written
records need a validated index before future replay or trust work can consume
them safely.

## Options Considered

1. Let downstream gates parse `.ai/memory/project.md` directly.
2. Promote append-only memory records directly into source identity trust.
3. Add a separate read-only memory index for append-only source identity blocks.

## Decision

Use option 3. Add a transport-free memory index that parses the writer's
append-only markdown blocks, validates their required fields and safety line,
and exposes records as review evidence only.

## Why This Won

The index closes the durable-memory visibility gap without confusing memory
presence with autonomous decision authority. It lets Product Layer preview show
what has been persisted while keeping model mutation, runtime trust mutation,
downstream priority mutation, queue mutation, and live console writes blocked.

## Rejected Alternatives

- Direct downstream parsing was rejected because each proposal family needs a
  validated evidence contract rather than file-format knowledge.
- Direct trust promotion was rejected because replay freshness, current source
  context, and operator review are still required before memory influences
  proposal priority.
- A separate storage dependency was rejected because append-only project memory
  is the current audited durable surface.

## Implementation Plan

1. Add a backend index module with strict block parsing and fail-closed
   validation.
2. Validate target path, draft ID prefix, source candidate/review IDs, trust
   state, reviewer, UTC timestamp, and exact runtime-authority line.
3. Wire the index into `main_pipeline_preview` as read-only evidence.
4. Register the index as `shadow_only` memory-learning evidence.
5. Add focused parser, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_index.py tests/test_source_identity_calibration_memory_writer.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: indexed memory could be mistaken for downstream trust authority.
Mitigation: records force `usable_for_downstream_priority=false` and safety
flags force `queue_mutation=false`, `model_mutation=false`,
`runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
`live_authority_granted=false`.

Rollback: remove the index call from `processing_pipeline_preview.py`, remove
the registry/capability rows, and keep appended source identity memory visible
only as raw project memory until a replacement index contract is reviewed.
