# ADR: Supervised WING Proof Plan

Date: 2026-06-04

## 1. Context

The current artifact report and evidence bridge show no physical promotion
candidates. To move toward supervised AutoMix safely, the project needs a
repeatable plan for the next physical WING proof runs.

## 2. Options considered

- Execute physical proof automatically.
- Keep the next proof as informal prose.
- Generate a read-only operator proof plan.

## 3. Decision

Add `supervised_wing_proof_plan`. It converts promotion evidence rows into
bounded physical proof steps for fader/gain, including command tuples,
prerequisites, abort conditions, and expected artifacts.

## 4. Why this won

It turns the absence of physical candidates into concrete next actions while
preserving operator authority. The plan does not execute commands, does not
connect to any mixer transport, and does not widen live policy.

## 5. Rejected alternatives

- Automatic execution was rejected because live WING proof requires explicit
  operator approval.
- Prose-only planning was rejected because command shape and abort conditions
  should be tested.
- Positive gain proof was rejected because gain-up remains unsafe without
  separate headroom/bleed and operator-authored reason proof.

## 6. Implementation plan

1. Add a pure Python proof plan builder.
2. Generate negative fader/gain proof commands for the existing physical runner.
3. Embed the plan in proof artifact reports.
4. Register the plan as shadow-only in the processing registry.
5. Add tests for no execution authority, skipped candidates, invalid artifacts,
   and command shape.

## 7. Test plan

- Compile touched Python files.
- Run focused proof/evidence/report/registry tests.
- Run broader readiness and no-WING boundary tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: generated commands are mistaken for permission. The plan marks
`execution_allowed_by_plan=false`, `live_write=false`, and
`operator_required=true`. Rollback is to remove the plan builder and report
embedding; no live path is introduced.
