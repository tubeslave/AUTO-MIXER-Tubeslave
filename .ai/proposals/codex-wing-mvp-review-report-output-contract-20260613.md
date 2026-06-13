# Codex Proposal: WING MVP Review Report Output Contract

## Short Thesis

The WING read-only review report paths should be shared contract values and
validated in the handoff packet artifact.

## What I Understood

The current MVP is still gated on real WING read-only proof. The operator uses
the handoff packet to generate an artifact review report and a decision review
report before any readiness update or supervised proof. Duplicated report-output
paths can make the operator inspect stale files.

## Proposed Solution

- Add shared contract constants for the artifact-review report output and
  decision-review report output.
- Make handoff/no-WING command builders consume those constants.
- Add handoff packet match flags for both report paths.
- Validate packet fields, declared flags, and command `--output` values in
  no-WING handoff artifact schema.
- Surface the fields in no-WING JSON and Markdown summaries.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-review-report-output-contract-20260613.md`
- `.ai/proposals/codex-wing-mvp-review-report-output-contract-20260613.md`
- `Docs/adr/wing-mvp-review-report-output-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_REPORT_OUTPUT_CONTRACT_20260613.md`

## Alternatives Considered

- Keep literals in each tool. Rejected because path drift is exactly the failure
  being removed from the handoff chain.
- Only add constants without validation. Rejected because stale handoff packets
  could still pass.
- Change review tool private defaults. Rejected for this step because the
  operator handoff path is the contract surface; tool-local defaults can remain
  separate unless promoted by a later decision.

## Risks

The main risk is stale tests or downstream summary consumers expecting only the
old command fields. Runtime risk is low: no hardware is contacted and no live
authority changes.

## Test Plan

- Compile touched modules.
- Run focused handoff, no-WING, Project Director, direction guard, and rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Where The Other Agent May Disagree

Kimi may prefer also changing the review tools' private default output paths.
That can be considered later, but the immediate safety boundary is the handoff
packet and release gate that operators use for WING proof.
