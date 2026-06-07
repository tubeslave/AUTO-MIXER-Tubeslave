# Codex Proposal: Supervised Proof Evaluator Physical Status Gate - 2026-06-07

## Short thesis

Physical supervised proof artifacts must prove both physical source and physical
status. A cache status must block promotion even when the artifact labels the
source as physical.

## What I understood

The runner already fails closed for live `--execute`, but the evaluator is the
downstream proof boundary for local summaries and promotion evidence. It needs
the same physical-status rule so stale or hand-edited artifacts cannot become
supervised candidates.

## Proposed solution

Add a physical-only status set to
`backend/supervised_proof_artifact_evaluator.py` and append physical blockers
when pre-read, apply, or rollback status is not
`verified_from_physical_readback`.

Add a no-WING static policy check for the evaluator source, plus focused unit
and report tests that keep `verified_from_cache` blocked in physical proof mode.

## Likely files to touch

- `backend/supervised_proof_artifact_evaluator.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_report.py`
- `tests/test_no_wing_release_check.py`
- council/report docs

## Alternatives considered

- Trust runner output only. Rejected because reports can inspect stored or
  legacy artifacts and must independently fail closed.
- Require physical status for every dry-run. Rejected because dry-run remains
  non-mutating rehearsal evidence.

## Risks

Older physical-looking artifacts with cache status will now be blocked. That is
intentional; they should be rerun under the current physical schema.

## Test plan

- Compile changed Python modules.
- Run focused evaluator/report/no-WING tests.
- Run quick no-WING release check.
- Run direction guard, memory lint, diff check, and full pytest before landing.

## Where the other agent may disagree

Kimi may ask for artifact schema migration. Blocking ambiguous/cache-status
physical artifacts is enough for this slice; a migration can be a separate
operator-facing cleanup if old reports need normalization.
