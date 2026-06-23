# ADR: WING MVP handoff prestart summary actual metadata

## Context

The WING read-only handoff packet references a no-WING prestart `summary.json`.
That summary must be passed before a real read-only preflight can proceed. The
packet previously rendered expected prestart metadata constants even when an
existing file was provided.

## Options considered

1. Keep rendering expected constants only.
2. Fail the packet builder when the referenced prestart summary is invalid.
3. Preserve placeholder defaults when no file exists, but render actual file
   metadata when a file is present.

## Decision

Use option 3.

## Why this won

It keeps the builder non-transport and report-only while preventing a failed or
environment-blocked no-WING summary from being shown as `passed` in the
operator packet.

## Rejected alternatives

Expected constants only can hide real evidence drift. Builder hard-failure
would duplicate downstream validation and make offline packet generation less
diagnostic.

## Implementation plan

- Add a local JSON-object loader for existing prestart summary files.
- Copy actual top-level prestart metadata and backend check status/detail flags
  into the packet when present.
- Copy environment-blocked check details into the packet and markdown.
- Add tests for valid and environment-blocked summaries.

## Test plan

- `tests/test_wing_readonly_preflight_handoff_packet.py`
- Focused no-WING/Project Director/WING MVP rails.
- Full standard suite.

## Risks and rollback

Risk: downstream consumers may need to handle `None` if a malformed real file
is supplied. That is safer than replacing unknown metadata with expected
values. Rollback is to remove the actual metadata loader and restore expected
constants only.
