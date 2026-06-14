# Proposal: Enforce handoff operator sequence cardinality

## Short thesis

Treat the WING handoff operator sequence as an exact contract, not only a
minimum-length checklist.

## What I understood

The current first blocker is still WING read-only connection proof. The no-WING
handoff packet guides the operator into the hardware run, so extra steps before
the supervised gate can weaken the intended sequence even when the structured
gate is updated to match the shifted index.

## Proposed solution

- Add a shared handoff operator sequence count derived from the supervised
  prestart gate index.
- In `tools/no_wing_release_check.py`, fail closed unless the sequence count is
  exactly the contract count.
- Fail closed unless the supervised prestart gate appears at the contract index.
- Render expected count/index and match flags in no-WING summary output.
- Add a negative test for an extra step before the supervised gate with the
  structured gate updated to the shifted index.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING MVP rails, ADR, and report docs.

## Alternatives considered

- Keep relying on "gate is final". Rejected because a shifted final gate can
  still be final.
- Check only count. Rejected because the exact supervised gate index is the
  operator-visible boundary before proof planning.

## Risks

The change adds another strict packet invariant. It is limited to offline
handoff validation and can be rolled back without touching live WING behavior.

## Test plan

- Focused no-WING validator tests.
- Focused WING rail and Project Director tests after docs update.
- no-WING quick gate.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer allowing extra operator notes. I would keep notes outside the
operator sequence because this list is now a contract for the hardware handoff
order.
