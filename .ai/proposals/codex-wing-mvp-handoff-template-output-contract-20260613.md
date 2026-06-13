# Codex Proposal: WING MVP Handoff Template Output Contract

## Short Thesis

The operator-facing WING handoff packet should carry the same pending-template
output path contract that no-WING validates internally.

## What I Understood

The current MVP is blocked on real WING read-only proof. The operator must use
the handoff packet to run prestart checks, capture read-only artifacts, and then
write a pending review decision template. If the handoff packet and review
command disagree about that template path, the manual review can be pointed at a
stale file without granting live authority but still damaging the proof chain.

## Proposed Solution

- Add `decision_template_output_matches_contract` to the handoff packet.
- Validate packet `decision_template_output` against the shared contract in
  no-WING handoff artifact schema.
- Validate the review command's `--decision-template-output` value matches the
  packet field.
- Surface those fields in no-WING JSON and Markdown summaries.
- Add focused tests for happy path and mismatch path.

## Likely Files To Touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-handoff-template-output-contract-20260613.md`
- `.ai/proposals/codex-wing-mvp-handoff-template-output-contract-20260613.md`
- `Docs/adr/wing-mvp-handoff-template-output-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_TEMPLATE_OUTPUT_CONTRACT_20260613.md`

## Alternatives Considered

- Rely on no-WING evidence summary only. Rejected because the handoff packet is
  the operator-visible artifact.
- Only render the path in Markdown. Rejected because tests and JSON consumers
  need a structured contract-match field.
- Change the public path. Rejected because the path is already the shared
  contract and should remain stable.

## Risks

The main risk is stale fixtures in handoff artifact tests. Runtime risk is low:
the change does not contact hardware, change defaults, or grant live authority.

## Test Plan

- Compile touched tools.
- Run focused handoff, no-WING, Project Director, direction guard, and rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Where The Other Agent May Disagree

Kimi may prefer putting the contract check only in the packet builder. That is
not enough: no-WING must reject stale or manually modified packets before any
operator-facing readiness update.
