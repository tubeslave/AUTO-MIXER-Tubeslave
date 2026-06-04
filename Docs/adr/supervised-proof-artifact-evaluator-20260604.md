# ADR: Supervised Proof Artifact Evaluator

Date: 2026-06-04

## 1. Context

Autonomy readiness now has a fail-closed snapshot, but promotion decisions need
artifact-level evidence. Existing runners can produce dry-run and physical
summaries, yet there was no reusable Product Layer evaluator to distinguish
dry-run readiness from physical supervised promotion readiness.

## 2. Options considered

- Promote gain from the existing dry-run proof.
- Add checks directly inside each proof runner only.
- Add a read-only proof artifact evaluator.

## 3. Decision

Add `supervised_proof_artifact_evaluator` as a shadow/evaluation method. It
normalizes proof summaries, validates required evidence, and reports
`dry_run_ready`, `supervised_candidate`, or `blocked`.

## 4. Why this won

The evaluator keeps proof requirements explicit without adding live authority.
Dry-run artifacts can be useful, but physical promotion requires physical
pre-read, apply readback, rollback readback, rollback proof, emergency-stop
proof, blocked follow-up proof, operator approval, and idempotency evidence.

## 5. Rejected alternatives

- Direct gain promotion was rejected because physical WING proof is missing.
- Runner-only checks were rejected because readiness/reporting surfaces need a
  common evaluator.
- Docs-only proof policy was rejected because tests should enforce fail-closed
  behavior.

## 6. Implementation plan

1. Add a transport-free evaluator module.
2. Support normalized dataclass inputs and runner-style summary dictionaries.
3. Expose snapshot output for reporting.
4. Register the evaluator as shadow-only in the processing registry.
5. Add tests for dry-run, physical, missing-proof, unsupported-family, and
   autonomous-ready invariants.

## 7. Test plan

- Compile touched Python modules.
- Run focused evaluator and processing registry tests.
- Run broader promotion/read/write/safety boundary tests.
- Scan for live transport and direct write tokens.
- Smoke-check that dry-run proof is not promotable when physical proof is
  required and no artifact is autonomous-ready.

## 8. Risks and rollback

Risk: proof status is mistaken for permission. The evaluator sets
`live_write=false`, never marks Full Autonomous ready, and only reports
promotion candidates when physical proof evidence is complete. Rollback is to
remove the registry row and evaluator module; no live write path is introduced.
