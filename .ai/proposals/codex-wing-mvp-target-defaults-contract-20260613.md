# Proposal: WING MVP Target Defaults Contract

## Short thesis

Centralize WING read-only preflight target defaults in
`backend/wing_readonly_preflight_contract.py` and reject stale plan, capture, or
handoff command targets before a real WING session.

## What I understood

The current supervised WING Rack MVP is still blocked by real read-only
connection proof. The operator-facing preflight flow depends on matching target
values across expected-plan generation, capture command text, and the handoff
packet. Duplicated literals can silently drift and leave the operator with an
inconsistent run recipe.

## Proposed solution

- Add shared constants for the read-only runner URL, channel set, value types,
  and capture artifact root.
- Make the handoff packet builder consume those constants for its defaults.
- Make no-WING validation compare plan, capture, and handoff target defaults
  against the shared contract.
- Emit detail and summary fields that show the expected defaults and each
  command surface.
- Fail closed on stale target defaults.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Keep literals duplicated. Rejected because URL or artifact-root drift would be
  hard to spot before the real preflight.
- Only update the handoff builder. Rejected because no-WING must also prove the
  command bundle is internally consistent.

## Risks

- Existing downstream summaries could expect fewer fields. This is low risk
  because the change adds fields and preserves current values.
- A stale fixture could fail after the stricter validator. That is the intended
  fail-closed behavior.

## Test plan

- Compile touched modules.
- Run focused handoff/no-WING/director/guard/rails tests.
- Run no-WING quick with no hardware and no backend writes.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer not adding another set of constants for stable defaults. The
counterpoint is that these defaults are part of the real WING proof recipe, so
the contract should fail closed before an operator follows stale command text.
