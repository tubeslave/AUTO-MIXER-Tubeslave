# read-surface-evidence-summary-detail-20260605

## Context

The no-WING release schema now validates read-surface fixture summary drift,
declared fixture paths, pytest function names, and row contract/gap consistency.
Those details are present in the internal check result, but the top-level
summary only shows broad surface counts and physical-missing IDs.

## Goal

Expose the declaration and consistency proof facts in `summary.json` and
`summary.md` so operators and Product Layer readers can inspect the evidence
without parsing the full `results` array.

## Non-goals

- Do not add a new validator.
- Do not execute fixtures from the summary writer.
- Do not mark read surfaces as physically proven.
- Do not widen live write authority.

## Safety constraints

- Summary facts must stay read-only and non-autonomous.
- Physical readback gaps remain visible.
- Cached DTO fixture coverage remains partial until physical proof exists.

## Expected evidence

- `read_surface_fixture_evidence` summary includes declared fixture/test counts.
- The summary states that declaration validation, summary recomputation, and
  contract/gap consistency are enforced.
- Markdown output exposes the same facts.
- Existing no-WING quick release check remains `live_write=false`.
