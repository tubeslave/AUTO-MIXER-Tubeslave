# read-surface-summary-drift-gate-20260605

## Context

The no-WING release check trusts `read_surface_fixture_evidence` as a read-only
coverage map for console read surfaces. Its summary is useful for release
decisions, but release gates should verify that summary counts match the
underlying surface rows.

## Goal

Make the no-WING read-surface fixture evidence schema fail closed when summary
counts, status counts, or physical-missing IDs drift from the row data.

## Non-goals

- Do not generate fixtures.
- Do not query WING hardware.
- Do not mark any read surface fully autonomous.
- Do not widen live write authority.

## Safety constraints

- The evidence snapshot remains read-only and non-mutating.
- Physical readback gaps must stay visible.
- Malformed summary metadata should fail validation instead of crashing the
  release check.

## Expected evidence

- Validator recomputes counts from `surfaces`.
- Validator rejects summary/row drift and unknown row statuses.
- Unit tests cover corrupted summary metadata.
- no-WING release check still reports read-only, no-live-write status.
