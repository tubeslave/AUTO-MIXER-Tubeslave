# ADR: Supervised Proof Artifact Report Gate

Date: 2026-06-04

## 1. Context

The project needs a credible path from proposal-only and dry-run behavior to
supervised physical console control. A proof artifact evaluator now exists, but
proof summaries still need to be discovered and reported consistently before
any safety coverage row is updated.

## 2. Options considered

- Keep proof review manual.
- Treat every artifact summary as proof evidence.
- Add an offline report gate for supervised proof artifacts.

## 3. Decision

Add `tools/evaluate_supervised_proof_artifacts.py`. It reads local
`summary.json` files, evaluates proof-like fader/gain summaries, ignores
unrelated summaries, and writes a report with dry-run, physical-candidate,
blocked, invalid, and ignored categories.

## 4. Why this won

This creates a repeatable proof-review step without adding live authority. It
lets no-WING and WING validation artifacts be audited with the same evidence
contract while keeping dry-run proof separate from physical promotion.

## 5. Rejected alternatives

- Manual-only review was rejected because promotion evidence should be
  repeatable and machine-readable.
- Treating all summaries as proof was rejected because the artifact tree
  contains unrelated release, telemetry, and offline validation summaries.
- Requiring physical candidates by default was rejected because no-WING runs
  should still produce useful reports.

## 6. Implementation plan

1. Add a transport-free report tool.
2. Discover local summary artifacts recursively.
3. Ignore unrelated summaries and fail closed on invalid proof-like evidence.
4. Add exit-code gates for minimum physical candidates and blocked artifacts.
5. Add tests and include the command in no-WING release check coverage.

## 7. Test plan

- Compile touched Python files.
- Run report/evaluator/no-WING focused tests.
- Run readiness and write-surface boundary tests.
- Run the standard full pytest suite.

## 8. Risks and rollback

Risk: the report is mistaken for permission. The report sets
`live_write=false`, `live_widening_allowed=false`, and keeps autonomous-ready
ids empty. Rollback is to remove the report command and tests; no live write
path is introduced.
