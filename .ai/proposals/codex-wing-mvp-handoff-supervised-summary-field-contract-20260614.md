# Codex Proposal: WING MVP Handoff Supervised Summary Field Contract

## Short thesis

Bind the handoff packet to the supervised proof-plan summary field contract
before no-WING recomputes it.

## What I understood

The active blocker remains `WING read-only connection proof`. The current
first command requires the no-WING handoff summary to show both the required
supervised proof-plan summary fields and the expected canonical field list with
a true match verdict. The packet builder declared only the raw required list.

## Proposed solution

Add `expected_required_supervised_proof_plan_summary_fields` and
`required_supervised_proof_plan_summary_fields_match_contract` to
`tools/build_wing_readonly_preflight_handoff_packet.py`. Render both in packet
Markdown and make `tools/no_wing_release_check.py` fail closed when either
packet-declared field is stale.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, report, and index files

## Alternatives considered

Keeping the field as no-WING-summary-only preserves the final validation
outcome but leaves the handoff packet less self-contained. A larger schema
object was deferred because the existing packet already uses flat declared
self-check fields.

## Risks

The added fields are redundant with no-WING recomputation. That redundancy is
intentional: the packet declares its state, and no-WING remains the authority
that rejects stale declarations.

## Test plan

Run focused handoff packet and no-WING validator tests, compile the changed
tools, generate a handoff packet, run Direction Guard, run no-WING quick
fail-closed, run memory lint, `git diff --check`, and the standard full pytest
command.

## Where the other agent may disagree

Kimi may argue that the no-WING summary already proves the contract. The
counterpoint is operator handoff quality: the packet should carry the same
declared contract fields that the gate later verifies.
