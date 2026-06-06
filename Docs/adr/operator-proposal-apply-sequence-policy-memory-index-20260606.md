# ADR: operator-proposal-apply-sequence-policy-memory-index-20260606

## Context

The supervised operator proposal apply sequence can now produce audited,
operator-acknowledged append-only memory records. The project goal requires
long-term learning, but appended memory must not become live execution authority
without separate review gates.

## Options considered

1. Feed appended sequence memory directly into policy or execution.
2. Leave sequence memory as plain markdown with no structured read path.
3. Add a read-only sequence policy memory index.

## Decision

Use option 3. Add a transport-free memory index that parses appended supervised
sequence memory blocks as review evidence only.

## Why this won

The index makes durable supervised-learning evidence visible to the Product
Layer and registry while preserving the existing safety boundary. It advances
long-term memory without creating queue, model, policy, runtime-memory, or live
console authority.

## Rejected alternatives

- Direct policy/execution influence was rejected because memory text is not a
  trust promotion gate.
- Plain markdown only was rejected because future learning work needs a
  deterministic, testable evidence shape.

## Implementation plan

1. Add a standalone parser/builder for `### Supervised sequence policy memory`
   blocks.
2. Validate required fields, fader ceiling, readback/proof status, and runtime
   authority line.
3. Register the method as `shadow_only` memory-learning evidence.
4. Expose it in main pipeline preview as an adapter and payload.
5. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_index.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_index.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: forged memory records imply live authority. Mitigation: malformed or unsafe
records return replacement-required status and zero indexed records.

Rollback: remove the index module, registry/profile/matrix references, preview
payload, tests, ADR, and project-memory lesson.
