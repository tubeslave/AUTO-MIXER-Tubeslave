# ADR: WING MVP Value Types Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Read-only preflight commands are bounded to `fader`, `gain`, and `pan`. The
default command value-types string was shared, but runner and handoff validators
still carried local allow-lists.

2026-06-16 clarification: the read-only value-types allow-list is not the same
as supervised live proof authority. Read-only preflight may inspect `pan`, but
the next supervised live proof plan must stay limited to `fader` and `gain`.
Operator-facing no-WING summaries must make that live boundary visible and
fail closed if the Markdown surface shows a stale or widened live allow-list.

## Options considered

1. Keep local allow-lists in runner and handoff code.
2. Move the supported value-types allow-list into the shared preflight contract
   and validate runner support in no-WING.
3. Only derive defaults from the contract and leave validators local.

## Decision

Use option 2.

For supervised proof planning, keep a separate shared
`allowed_supervised_live_value_types` boundary derived from the preflight
contract but limited to `fader` and `gain`. Surface it in the proof-plan summary
and no-WING operator Markdown as expected/actual/match fields.

## Why this won

The supported value types are part of the real WING proof boundary. A future
change must not accidentally widen or narrow the runner/handoff validation scope
without passing through the shared contract and no-WING prestart gate.

## Rejected alternatives

- Option 1 allows command defaults and tool validators to drift.
- Option 3 centralizes display defaults but not the actual allowed request
  surface.

## Implementation plan

- Add shared supported value-types tuple to the preflight contract.
- Derive the default comma-separated value-types string from that tuple.
- Use the shared tuple in runner and handoff `SUPPORTED_VALUE_TYPES`.
- Make no-WING compare runner-supported value types to the shared tuple.
- Add shared supervised live proof value-types limited to `fader` and `gain`.
- Make no-WING reject proof plans whose `planned_value_types` include `pan` or
  any other value type outside the supervised live proof boundary.
- Make no-WING operator Markdown compare expected and actual supervised live
  value types so stale surfaces fail closed.
- Add focused tests for happy path and stale runner-supported values.

## Test plan

- Compile touched modules.
- Focused runner, handoff, and no-WING tests.
- no-WING quick on an unused backend port.
- Focused proof-plan/no-WING tests covering stale or widened supervised live
  value-type surfaces.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter prestart validation surfacing a stale local allow-list.
That is intended. Rollback would restore local sets, but would reopen drift
between the proof contract and the tools used for real read-only preflight.

The supervised-live clarification has a narrower rollback: remove the
`allowed_supervised_live_value_types` summary/Markdown checks while keeping the
read-only `fader`, `gain`, and `pan` preflight contract. That rollback should
only be used if the operator surface itself is wrong, not to widen live writes.
