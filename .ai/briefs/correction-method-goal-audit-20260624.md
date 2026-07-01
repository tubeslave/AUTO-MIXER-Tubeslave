# Correction Method Goal Audit

Task id: `correction-method-goal-audit-20260624`

## Thesis

The correction-method work needs a completion audit surface that proves offline
proposal-layer coverage without implying WING readiness or live authority.

## What I Understood

- The active user goal is to develop Automixer correction methods without a
  real mixer in offline, dry-run, proposal-only and read-only mode.
- The covered method families are gain/trim, fader/balance, EQ, compressor,
  gate/de-esser, FX/sends, pan/stereo, routing/groups/scenes read-only, and
  master/output proposal-only.
- Every method needs evidence, problem statement, proposed correction,
  confidence, bounded delta, safety blockers, applicable
  headroom/feedback/true-peak checks, operator-readable explanation, rollback,
  future WING readback contract and tests.
- Passing offline coverage must not close WING readiness or grant live apply,
  queue import, routing, scene, mute, EQ, compressor, gate or FX write authority.

## Proposed Solution

- Keep the existing operator review packet as the detailed method surface.
- Add and use a dedicated audit artifact that checks objective-family coverage,
  method-card coverage, operator surfaces, and authority-denial flags.
- Record the decision and report so future review does not mistake offline
  coverage for concert readiness.

## Likely Files Touched

- `tools/audit_correction_method_goal.py`
- `tests/test_processing_proposal_contracts.py`
- `Docs/adr/correction-method-goal-audit-20260624.md`
- `Docs/reports/AUTOMIXER_CORRECTION_METHOD_GOAL_AUDIT_20260624.md`

## Alternatives Considered

- Rely only on the large operator review export.
- Wait for real WING proof before documenting offline method coverage.
- Mark the full project complete from green tests.

## Risks

- The audit could be overread as live readiness.
- Generated artifacts can drift from code if they are not regenerated.
- The current worktree is dirty and includes previous uncommitted changes.

## Test Plan

- Compile changed Python files.
- Run focused correction-method goal audit tests.
- Run the full processing proposal contract tests.
- Run the full pytest suite before merge.
- Run `git diff --check`.

## Where Another Agent May Disagree

Another agent may ask to move the audit into the no-WING gate. That should be a
separate integration step; this change keeps the correction-method audit
focused on proposal-layer coverage only.
