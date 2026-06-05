# read-surface-contract-gap-consistency-gate-20260605

## Context

The no-WING release gate now validates read-surface fixture paths and declared
pytest functions. The next evidence weakness is row-level consistency: a row can
claim a fixture status that does not match its covered contracts or remaining
gaps.

## Goal

Make `read_surface_fixture_evidence_schema` fail closed when row status,
`covered_contracts`, and `remaining_gaps` contradict each other.

## Non-goals

- Do not mark any row as physical proof complete.
- Do not add live mixer reads or writes.
- Do not execute fixtures from the schema validator.

## Safety constraints

- `partial_fixture_evidence` must keep physical readback gaps visible.
- `fixture_evidence_present` must require explicit physical readback coverage
  and no physical-missing gap.
- `missing_fixture_evidence` must not carry stale contracts or declarations.

## Expected evidence

- Release schema validates `surface_id`, `target_console_surface`,
  `covered_contracts`, and `remaining_gaps`.
- Corrupted status/contract/gap rows fail with explicit violation IDs.
- no-WING quick release check still passes with read-only, non-autonomous
  evidence.
