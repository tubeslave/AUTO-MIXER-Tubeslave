# Brief: WING MVP Handoff Proof Plan Artifact Root Summary Fields

## Task

Require the WING read-only handoff packet `required_supervised_proof_plan_summary_fields` contract to include accepted-candidate artifact-root proof-plan diagnostics.

## Context

The supervised proof plan and no-WING validation now carry accepted-candidate `artifact_root` diagnostics. The handoff packet still needs to advertise those fields as required before supervised proof planning, otherwise stale handoff summaries can look complete while omitting the downstream root trace.

## Constraints

- No live mixer transport.
- No live-write authority.
- No positive gain or fader movement.
- Keep the first WING MVP blocker open until a real WING read-only candidate is captured and reviewed.
- Use the existing shared contract constant instead of duplicating field lists.

## Done When

- `required_supervised_proof_plan_summary_fields` includes the accepted-candidate artifact-root proof-plan fields.
- Handoff/no-WING tests assert the root fields are in the required field contract.
- Current command rails mention the root fields as part of the handoff-required contract.
- The decision is recorded in ADR/report docs and indexed.
