# Proposal: Enforce handoff prestart summary SHA availability

## Short thesis

Require the no-WING handoff packet to say the prestart summary digest is
available whenever it carries that digest.

## What I understood

The WING MVP remains on the read-only connection proof. The current handoff
packet now binds the prestart summary by SHA256, but a stale or hand-edited
packet could keep the digest while falsely marking availability as false.

## Proposed solution

- In `tools/no_wing_release_check.py`, compute whether the packet digest is
  present and compare it to `backend_port_prestart_summary_sha256_available`.
- Add a fail-closed violation for mismatch.
- Surface both availability and availability-match fields in no-WING summary
  JSON/Markdown.
- Add tests for the valid case and the mismatch case.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING MVP rail docs, ADR, report, and Project Director tests.

## Alternatives considered

- Ignore the availability flag because digest matching is stronger. Rejected:
  the flag is operator-facing state and should not contradict the digest.
- Remove the flag. Rejected because placeholder packet generation benefits from
  an explicit unavailable state.

## Risks

The main risk is adding another packet invariant. It is limited to no-WING
handoff artifact validation and keeps standalone placeholder packets unchanged.

## Test plan

- Focused `tests/test_no_wing_release_check.py`.
- Focused WING rail and Project Director tests.
- no-WING quick gate.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may consider this redundant with digest equality. I would keep it because
operator-visible packet state should not be internally contradictory before a
hardware session.
