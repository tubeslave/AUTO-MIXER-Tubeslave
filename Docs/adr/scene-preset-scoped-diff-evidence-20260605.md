# ADR: scene-preset-scoped-diff-evidence-20260605

## Context

Scene/preset recommendations require target preview, scoped diff, rollback, and
scope evidence. Before this change those facts were separate booleans and
counts. The final autonomous goal needs a more structured evidence contract, but
scene and preset writes remain unsafe without hardware proof.

## Options considered

1. Add a structured read-only scoped-diff evidence payload.
2. Keep separate booleans and counts only.
3. Generate diffs by querying a live console from the proposal layer.

## Decision

Use option 1. `ScenePresetManagementAnalysisInput` accepts optional
`scoped_diff_evidence`, and the analyzer derives safe proposal evidence from it
when it is read-only and transport-free.

## Why this won

This creates a concrete target contract for future Product Layer diff DTOs while
preserving fail-closed behavior. A scoped diff can support proposal-only
recommendations; a live/query/mutating or broad diff blocks them.

## Rejected alternatives

Manual booleans were insufficient as a long-term contract. Live queries were
rejected because the proposal layer must not touch hardware.

## Implementation plan

1. Add optional scoped-diff evidence to the scene/preset analysis input.
2. Normalize diff payloads into availability, rollback, affected counts,
   destructive-scope, and safety flags.
3. Add blockers for live/query/mutating diff payloads.
4. Include normalized diff evidence in recommendation evidence.
5. Update registry/contract metadata and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/scene_preset_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py tests/test_scene_preset_management_proposal.py tests/test_processing_registry.py`
- Focused scene/preset proposal tests.
- Registry/contract tests.
- Standard project pytest command.

## Risks and rollback

Risk: scoped-diff evidence may be confused with apply authority. The proposal
payload remains non-queueable and non-applicable, with no live write kinds.

Rollback is removing the evidence field, normalizer, tests, metadata updates,
and this ADR. No live write path is introduced.
