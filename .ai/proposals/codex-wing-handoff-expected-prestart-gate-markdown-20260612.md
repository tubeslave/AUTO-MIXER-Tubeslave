# Codex Proposal: WING Handoff Expected Prestart Gate Markdown

## Short thesis

Add the expected supervised prestart gate to the handoff packet itself and make
the validator reject drift between the declared expected gate and the computed
contract.

## What I understood

The current MVP is still blocked on real WING read-only proof. The handoff
packet already carries `supervised_prestart_gate`, and the no-WING summary
computes `expected_supervised_prestart_gate`, but the operator handoff Markdown
does not show the expected contract beside the actual one.

## Proposed solution

- Add `expected_supervised_prestart_gate` to the handoff packet JSON.
- Render an `Expected Supervised Prestart Gate` section in the Markdown packet.
- Add no-WING validation that rejects mismatched declared expected gates.
- Add focused regression coverage for builder output, Markdown, and validator
  rejection.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Leave expected gate only in no-WING summary. Rejected because operators may
  review the handoff packet directly.
- Render prose only. Rejected because this boundary should remain
  machine-readable and diffable.

## Risks

- Slightly more verbose Markdown.
- Test fixture churn if the prestart gate contract changes later.

## Test plan

- Focused handoff/no-WING tests.
- Adjacent WING MVP rail tests.
- `git diff --check`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may argue that the computed expected gate in no-WING summary is enough.
I would keep both because the packet is the operator-facing artifact used
before the real hardware step.
