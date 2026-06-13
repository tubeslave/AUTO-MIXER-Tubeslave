# ADR: WING MVP Target Defaults Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before that proof, offline tooling prepares the expected plan, capture command,
and handoff packet. Those surfaces must describe the same read-only WING target
without granting live authority.

Target defaults for URL, channels, value types, and artifact root were duplicated
between handoff and no-WING tooling.

## Options considered

1. Keep the target defaults duplicated.
2. Move the defaults into the shared WING read-only preflight contract and
   validate all command surfaces against it.
3. Only update the handoff packet builder defaults.

## Decision

Use option 2.

## Why this won

The target defaults form part of the operator's real WING preflight recipe.
Centralizing them in `backend/wing_readonly_preflight_contract.py` keeps the
plan, capture, and handoff packet aligned and lets no-WING fail closed before
hardware is involved.

## Rejected alternatives

- Option 1 allows silent command drift.
- Option 3 leaves no-WING unable to reject stale target defaults across the full
  command bundle.

## Implementation plan

- Add shared constants for runner URL, runner channels, runner value types, and
  capture artifact root.
- Make the handoff packet builder use those constants as defaults.
- Make no-WING validation compare plan, capture, and handoff target defaults
  against the shared contract.
- Add detail, JSON summary, and Markdown summary fields for the target-default
  comparisons.
- Cover the happy path and stale handoff URL path in focused tests.

## Test plan

- Compile touched modules.
- Run focused handoff, no-WING, Project Director, direction guard, and WING MVP
  rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard command checks.
- Run `git diff --check`.
- Run `python3 tools/memory/memory_lint.py`.
- Run the full backend pytest suite.

## Risks and rollback

The main risk is stricter no-WING validation exposing stale command fixtures.
That is the intended behavior. Rollback restores duplicated literals, but it
reopens the drift path between operator-facing preflight commands.
