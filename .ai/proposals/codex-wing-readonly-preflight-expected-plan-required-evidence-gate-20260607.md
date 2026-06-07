# Proposal: Evidence Gate For Required Expected Plan

## Short Thesis

WING read-only evidence intake should reject artifacts that do not explicitly
prove the runner required an expected plan for the real run.

## What I Understood

The current MVP is blocked on real WING read-only proof. Since the runner now
fails closed without `--expected-plan`, accepted proof artifacts should preserve
that fact in their structured evidence.

## Proposed Solution

Add `expected_plan_required_for_real_run` to the evidence validator's required
truthy fields. Update valid fixtures and add a regression assertion to the
existing missing-expected-plan-link test.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- focused WING preflight evidence/review/autonomy tests
- `.ai/briefs/wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-expected-plan-required-evidence-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_REQUIRED_EVIDENCE_GATE_20260607.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Rely only on runner behavior. Rejected because evidence intake must defend
  against stale, hand-written, or copied summaries.
- Treat the field as optional. Rejected because the first hardware proof needs
  explicit operator-visible evidence.

## Risks

- Older preflight artifacts without the field will be invalidated. That is
  intended for current proof quality.

## Test Plan

- Focused evidence/review/autonomy tests.
- no-WING quick gate.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue path/SHA/match already imply the requirement. The explicit flag
is still valuable because it states the pre-transport policy in the artifact
itself.
