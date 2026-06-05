# read-surface-physical-readback-capture-plan-20260605

## Context

Read-surface fixture evidence and remediation summaries now identify the next
physical readback gap, but the operator still needs a concrete non-executing
checklist before any future hardware artifact can be reviewed.

The current worktree already contains a `read_surface_physical_readback_capture_plan`
slice that builds such a checklist from fixture and remediation metadata.

## Goal

Keep the capture plan visible through registry, Product Layer, no-WING release
summary, and frontend read-only status rows while making its safety boundary
explicit: it is operator-required planning evidence, not an executor.

## Non-goals

- Do not query WING or dLive hardware.
- Do not run physical readback capture.
- Do not execute tests from the capture-plan module itself.
- Do not grant live write, live authority, or Full Autonomous readiness.

## Safety constraints

- `read_only=true`, `operator_required=true`, and `execution_allowed_by_plan=false`
  must remain explicit.
- `live_write=false`, `live_authority_granted=false`, and
  `full_autonomous_ready=false` must be serialized in release evidence.
- The plan must not import mixer clients, sockets, websocket transports, or
  supervised apply entrypoints.
- Any source claiming execution authority must fail validation or be normalized
  back to read-only evidence.

## Expected evidence

- Focused backend tests pass for the capture plan, no-WING release summary,
  Product Layer proof-quality normalization, processing registry, and capability
  matrix.
- Frontend adapter contract preserves the capture plan as a read-only status
  row.
- Full backend tests, no-WING quick check, memory lint, and diff hygiene pass.
