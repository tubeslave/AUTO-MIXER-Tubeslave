# WING Read-Only Source Path Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095 blocked reserved
no-WING/control-plane roots, M096 blocked declared summary/events path-shape
mismatches, and M097 blocked non-canonical WING artifact roots. A remaining
file-intake gap allowed a copied summary file to be loaded from
`.../stale-source/summary.json` while the JSON payload still declared
`summary_path` as the canonical `<artifact_root>/<run_id>/summary.json`.

## Options Considered

1. Leave source-file binding to manual operator review.
2. Reject file-loaded candidates when source path and declared `summary_path`
   differ.
3. Accept copied summaries if all other read-only proof fields are valid.

## Decision

Use option 2. When a candidate is loaded from disk, the source file path must
match the payload's declared `summary_path`; otherwise intake reports
`source_path_declared_summary_path_mismatch`.

## Why This Won

The first accepted-candidate boundary should not allow a stale or copied summary
file to masquerade as the reviewed run. Binding source path to declared
summary path makes the operator review target unambiguous before later decision
ledger gates run.

## Rejected Alternatives

Manual-only review was rejected because the automated intake can cheaply catch
the mismatch before a bad candidate looks ready. Accepting copied summaries was
rejected because it weakens the real WING artifact-location contract.

## Implementation Plan

- Compare file-loaded `summary_path` to the payload's declared `summary_path`.
- Add focused regression coverage for a copied stale-source summary.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- M098 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting manually copied summaries that still point back to
the original run directory. That is intended for M001: real proof must be
reviewed at the exact captured source path. Roll back by reverting this ADR, the
M098 plan/report/state entries, the test and the source-path guard. No live transport
or console state is affected.
