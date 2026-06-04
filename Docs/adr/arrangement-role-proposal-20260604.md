# ADR: Arrangement Role Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires automatic determination of instrument roles in the
arrangement. Existing classifier and agentic layers can infer source
information, but arrangement roles were still shadow-only. The live executor
remains limited to supervised fader/gain.

## 2. Options considered

- Keep arrangement roles shadow-only.
- Add a proposal-only arrangement-role evidence layer.
- Let inferred roles directly affect downstream console actions.

## 3. Decision

Add `arrangement_role_proposal` as a transport-free proposal-only method. Move
`identify_arrangement_roles` to `proposal_only` with an Assisted-mode ceiling.
The output is non-queueable and does not create a live write request.

## 4. Why this won

This advances the intelligence layer needed for professional mixing without
widening live authority. The method records source-type confidence, role
confidence, arrangement context, inter-channel context, reference evidence,
accepted-history evidence, persistence, ambiguity, and masking impact.

## 5. Rejected Alternatives

- Direct downstream authority was rejected because role mistakes can lead to bad
  fader, EQ, FX, or dynamics choices.
- Shadow-only status was rejected because it leaves no operator-review learning
  loop.
- Treating source type as arrangement role was rejected because a guitar can be
  lead, support, texture, or effect depending on section and context.

## 6. Implementation Plan

1. Add a pure Python arrangement-role analyzer.
2. Emit only non-queueable proposal payloads with no `requested_change`.
3. Register the method in processing metadata and proposal contracts.
4. Update the project-goal capability row.
5. Add tests for safe proposals, blockers, and live-boundary invariants.

## 7. Test Plan

- Compile touched Python modules.
- Run focused arrangement-role tests.
- Run broader proposal-boundary tests for registry and capability matrix.
- Run `git diff --check` and payload smoke checks.

## 8. Risks And Rollback

Risk: role proposals are treated as authoritative before replay learning. The
payload is non-queueable, has `can_apply_live=False`, includes unsupported
value-type blockers, and contains no requested change. Rollback is to remove the
method from registry/profile/matrix wiring; no direct live path is introduced.
