# ADR: Supervised Promotion Evidence Bridge

Date: 2026-06-04

## 1. Context

The repository has a supervised proof artifact evaluator and an offline report
tool. The next safety gap is connecting proof report outcomes to the safety
coverage rows that define whether fader/gain can be considered for supervised
live use.

## 2. Options considered

- Automatically update safety coverage rows from reports.
- Leave proof report interpretation manual.
- Add a read-only evidence bridge that produces manual review candidates.

## 3. Decision

Add `supervised_promotion_evidence_bridge`. It reads proof report rows and the
current safety matrix, then returns fader/gain evidence rows with manual
safety-row review candidacy, blockers, and next evidence.

## 4. Why this won

This keeps promotion evidence testable while preserving operator control. A
physical proof candidate can justify manual review, but the bridge never edits
the safety matrix, never widens live writes, and never marks Full Autonomous
ready.

## 5. Rejected alternatives

- Automatic safety updates were rejected because they would turn proof parsing
  into live authority.
- Manual-only interpretation was rejected because readiness claims should be
  backed by repeatable machine-readable evidence.
- Broad action-family support was rejected because EQ, dynamics, FX, routing,
  and scenes do not share the narrow supervised fader/gain executor.

## 6. Implementation plan

1. Add a pure Python evidence bridge.
2. Support fader and gain safety rows.
3. Embed bridge output into the proof artifact report.
4. Register the bridge as shadow-only evaluation.
5. Add tests for dry-run-only, physical candidate, blocked artifact, invalid
   summary, and no-live-widening behavior.

## 7. Test plan

- Compile touched Python files.
- Run focused proof/evidence/registry/no-WING tests.
- Run broader readiness and write-surface tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: evidence rows are mistaken for permission. The bridge sets
`automatic_safety_update_allowed=false`, `live_widening_allowed=false`, and
`full_autonomous_ready=false`. Rollback is to remove the bridge and registry
row; no live write path is introduced.
