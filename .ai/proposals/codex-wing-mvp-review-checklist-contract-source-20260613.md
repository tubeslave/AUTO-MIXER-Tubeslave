# Codex Proposal: WING MVP Review Checklist Contract Source

## Short Thesis

Move WING read-only manual-review checklist ids into the shared preflight
contract so the operator review packet, validator, and ledger cannot drift
before real hardware proof.

## What I Understood

The current MVP is still the supervised WING Rack proof, with
`WING read-only connection proof` first. The manual review checklist is an
operator safety gate: it verifies artifact paths, channel scope, session
readiness, physical readbacks, read-only event logs, and no live authority.

## Proposed Solution

Add `WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS` to
`backend/wing_readonly_preflight_contract.py`. Update
`backend/wing_readonly_preflight_review_decision.py` and
`tools/no_wing_release_check.py` to keep compatibility aliases from the shared
source. Update the handoff packet builder to consume the shared constant
directly.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- focused handoff/review/no-WING tests
- WING MVP report/ADR/index docs

## Alternatives Considered

Leaving the ids in the release checker preserves today's behavior but keeps the
handoff builder dependent on validator internals. Keeping duplicate constants in
review decision and no-WING code raises drift risk. The shared preflight
contract is the right boundary because the checklist defines operator review
semantics, not command execution.

## Risks

The main risk is changing checklist order or dropping a required operator
confirmation. Tests should compare exact generated packet values, no-WING
aliases, review-tool checklist ids, and review-decision aliases to the shared
contract.

## Test Plan

- Compile touched modules.
- Run focused review/handoff/no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Where The Other Agent May Disagree

Kimi may argue this is still not real hardware progress. The counterpoint is
that the checklist is an operator gate in the first real hardware handoff; a
single source reduces the chance of an unsafe or stale manual review packet.
