# Brief: operator proposal apply-sequence policy memory trust promotion queue-import promotion preflight-plan result consumer

## Context

The promotion queue-importer result consumer can verify applied queue-importer evidence and emit a queue-import preflight-plan request template. The next safe boundary is to verify the resulting read-only preflight plan before exposing the existing batch preflight handler.

## Task

Add a fail-closed verifier that consumes a verified promotion queue-importer result consumer plus a queue-import preflight-plan result. It should expose only a `preflight_operator_proposal_apply_batch` request template after the plan matches the applied importer provenance.

## Safety constraints

- Do not call mixer transport or manual console write paths.
- Do not run batch preflight inside the verifier.
- Do not reserve idempotency, mutate proposal status or order, mutate queues, mutate policy/runtime/model/memory state, or grant live authority.
- The emitted batch preflight request must keep `approved=false`; a separate operator action must set `approved=true` before the existing batch preflight handler can mark proposals ready.
- Forged plan summary, request, or spec authority must be rejected.

## Validation

- Focused unit tests for missing inputs, dry-run importer blocking, ready plan request availability, forged plan rejection, registry, and capability matrix.
- Product Layer dispatch test for applied queue import -> verifier -> preflight plan -> verifier -> batch preflight request.
- Compile, memory lint, diff check, and focused pytest.
