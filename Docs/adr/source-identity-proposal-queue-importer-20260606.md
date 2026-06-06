# ADR: source-identity-proposal-queue-importer-20260606

## Context

The source identity queue-ordering plan can emit future import candidates. The
project still lacks a controlled consumer that validates those candidates,
renders import specs, and can exercise the native operator proposal queue under
explicit acknowledgement.

## Options Considered

1. Import queue-ordering plan records automatically in Product Layer preview.
2. Keep queue import permanently out of backend code until UI work begins.
3. Add an operator-gated queue importer with Product Layer dry-run only.

## Decision

Use option 3. Add a queue importer that emits dry-run specs by default and
mutates only a caller-provided target queue when invoked with `apply=true` and
`operator_ack=true`.

## Why This Won

It advances autonomous decision routing into the real queue boundary while
preserving operator review, idempotent future UI work, and live-console safety.

## Rejected Alternatives

- Automatic Product Layer import was rejected because preview builders must stay
  non-mutating.
- Waiting for UI only was rejected because the backend trust boundary can be
  tested now without exposing a live write path.
- Importing skeletal rows was rejected because the importer should preserve
  source proposal payloads when queue mutation is explicitly requested.

## Implementation Plan

1. Add the backend source identity proposal queue importer module.
2. Validate queue-ordering plan envelope, summary, records, duplicate IDs, and
   safety flags.
3. Require matching source proposal payloads for import specs.
4. Emit dry-run import specs by default.
5. Require `operator_ack=true` and an explicit target queue for apply.
6. Wire Product Layer dry-run output, registry, capability matrix, and tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_queue_importer.py tests/test_source_identity_proposal_queue_ordering_plan.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_queue_importer.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: queue import could be mistaken for live apply. Mitigation: importer
results keep `live_write=false`, and console apply remains behind the existing
operator proposal approval and supervised manual write gate.

Risk: Product Layer accidentally mutates the queue. Mitigation: Product Layer
calls the importer with `apply=false` and no target queue.

Rollback: remove the queue importer call from `processing_pipeline_preview.py`,
remove registry/capability rows, and keep the queue-ordering plan as the
terminal source identity queue-boundary artifact.
