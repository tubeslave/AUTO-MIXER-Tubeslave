# ADR: mixing-station-trim-read-parity-20260605

## Context

The project uses `/ch/{N}/in/set/trim` as the canonical input trim key for
WING-style gain staging, SafeGain proposals, offline validation, and supervised
manual writes. Mixing Station is a supported mixer client, but its gain helper
used `/ch/{N}/gain`, creating a cache/readback mismatch with the manual gate.

## Options considered

1. Leave Mixing Station gain as `/ch/{N}/gain`.
2. Normalize Mixing Station gain/trim aliases to `/ch/{N}/in/set/trim`.
3. Add Mixing Station-specific alias handling inside the manual gate.

## Decision

Use option 2. Mixing Station input gain/trim cache, query, and write helpers
now use the same canonical trim address as the rest of the backend.

## Why this won

It keeps the supervised gate mixer-agnostic and makes cache freshness,
pre-read, readback, and rollback evidence comparable across WING, dLive,
offline validation, and Mixing Station.

## Rejected alternatives

Option 1 preserves fragmented state keys. Option 3 pushes client-specific
knowledge into the central safety gate and makes future proof evidence harder
to compare.

## Implementation plan

- Canonicalize `/ch/{N}/gain`, `/ch/{N}/trim`, and preamp aliases to
  `/ch/{N}/in/set/trim`.
- Map outgoing canonical trim to Mixing Station `ch.N.preamp.gain`.
- Timestamp incoming preamp gain cache updates on the canonical key.
- Add read-only `query_parameter()` for server readback compatibility.
- Add dummy-socket tests with no real network traffic.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_mixing_station_client.py -q`
- Focused manual gate/operator proposal tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: a specific Mixing Station-backed mixer may expose a different preamp gain
path. Rollback is removing the canonical alias mapping, `query_parameter()`,
initial trim scan, tests, and this ADR. Live authority is unchanged either way.
