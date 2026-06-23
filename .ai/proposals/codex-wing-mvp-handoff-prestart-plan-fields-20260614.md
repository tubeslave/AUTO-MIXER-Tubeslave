# Codex Proposal: WING MVP Handoff Prestart Plan Fields

## Short thesis

The handoff packet should carry the same prestart-owned required-false plan
fields that the downstream no-WING and candidate traces now require.

## What I understood

The active blocker is still WING read-only connection proof. Current rails
already require `plan_<field>: false` and `<field>_matches_plan: true` markers,
but `tools/build_wing_readonly_preflight_handoff_packet.py` only renders raw
required-false fields and match flags. It does not expose the `plan_<field>`
values directly at the handoff packet boundary.

## Proposed solution

Import the shared required-false plan/match constants into the handoff packet
builder, carry plan values from the referenced no-WING prestart summary when
present, compute handoff match verdicts from raw values versus those plan
values, default them only for placeholder/offline packet generation, and render
them in markdown. Add focused packet tests.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- Council/report docs for this slice.

## Alternatives considered

- Leave the fields only in no-WING summary artifacts. Rejected because the
  operator packet is the handoff surface and should not hide a stale or missing
  prestart plan field.
- Trust recorded match flags blindly. Rejected because the handoff boundary
  should detect raw-value or plan-value drift in the referenced summary.

## Risks

Older prestart summaries without these fields will surface `None` in handoff
packets instead of being silently normalized. That is fail-closed behavior for
operator review and downstream checks.

## Test plan

- Focused handoff packet tests.
- Focused no-WING release checks that consume the packet.
- Project Director summary sanity.
- Compile and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer a nested `prestart_plan` object. I would keep flat fields
because current reports, CLI summaries, and Project Director markers are flat.
