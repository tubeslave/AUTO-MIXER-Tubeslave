# Codex Proposal: WING Proof Plan Candidate Prestart Blocker Gate

## Short thesis

If a proof plan sees accepted source ids but cannot bind them to prestart-valid candidate rows, the no-WING release gate should require the precise candidate-prestart blocker.

## What I understood

The proof-plan builder already blocks physical steps when accepted source ids are not backed by valid `candidate_reviews` prestart fields. The release validator must enforce the same operator-facing reason, not only the generic manual-review state.

## Proposed solution

- Add a local no-WING release-check constant for `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.
- In `validate_supervised_proof_report_next_wing_plan`, require that blocker when accepted source ids exist and the expected candidate prestart contract is invalid.
- Add regression coverage for both missing-blocker failure and valid blocked state.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## Alternatives considered

- Trust the generic review-decision blocker. That hides the actual remediation: accepted source ids must be regenerated or reviewed with candidate prestart fields.
- Import the backend proof-plan blocker constant. The release-check module already keeps many local gate constants; a local string constant avoids an extra import boundary.

## Risks

- Older snapshots with invalid candidate prestart trace and generic blockers now fail release validation. This is intended fail-closed behavior at the operator handoff boundary.

## Test plan

- Focused no-WING next-plan tests.
- Affected WING/no-WING stack.
- Full standard pytest suite.
- Diff and memory lint.

## Where the other agent may disagree

Kimi may prefer sharing the blocker constant from the backend module. I would keep this release-check-local for now and consider a small shared constants module only if another boundary needs the same string.
