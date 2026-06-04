# ADR: Decision Evidence Packet Proposal Layer

Date: 2026-06-04

## 1. Context

The project goal requires decisions to be based on audio, psychoacoustics,
statistics, history, references, previous decisions, inter-channel context,
arrangement, and genre. Existing systems expose these as separate signals, but
`decision_basis` was still shadow-only. The live executor remains limited to
supervised fader/gain.

## 2. Options considered

- Keep decision basis shadow-only.
- Add a proposal-only decision evidence packet layer.
- Let evidence packets authorize live actions directly.

## 3. Decision

Add `decision_evidence_packet_proposal` as a transport-free proposal-only
method. Move `decision_basis` to `proposal_only` with an Assisted-mode ceiling.
The output is non-queueable and does not create a live write request.

## 4. Why this won

This creates a stable operator-review contract for proposal and rejected shadow
decisions without widening live authority. The method records evidence family
coverage, family confidence, critical missing evidence, readback, rollback,
true-peak headroom, feedback risk, latency, persistence, and conflicts.

## 5. Rejected Alternatives

- Direct live authority was rejected because evidence quality does not prove
  actuator safety.
- Shadow-only status was rejected because it leaves no unified evidence packet
  for operator review or replay learning.
- Requiring every evidence family was rejected because some contexts may lack a
  reference or project-history signal; critical evidence plus minimum coverage
  is the safer initial contract.

## 6. Implementation Plan

1. Add a pure Python decision evidence packet analyzer.
2. Emit only non-queueable packet payloads with no apply fields.
3. Register the method in processing metadata and proposal contracts.
4. Update the project-goal capability row.
5. Add tests for safe packets, blockers, and live-boundary invariants.

## 7. Test Plan

- Compile touched Python modules.
- Run focused decision evidence packet tests.
- Run broader proposal-boundary tests for registry and capability matrix.
- Run `git diff --check` and payload smoke checks.

## 8. Risks And Rollback

Risk: evidence packets are treated as live authority before replay learning. The
payload is non-queueable, has `can_apply_live=False`, includes unsupported
value-type blockers, and contains no apply request. Rollback is to remove the
method from registry/profile/matrix wiring; no direct live path is introduced.
