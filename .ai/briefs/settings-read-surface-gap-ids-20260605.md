# settings-read-surface-gap-ids-20260605

## Context

Product Layer read-surface proof quality already includes
`physical_fixture_missing_surface_ids`, but Settings only shows the total number
of physical readback gaps. Operators need the first missing surface visible
without opening artifact JSON.

## Goal

Expose the read-surface physical gap surface IDs in the existing read-only
Settings proof evidence path.

## Non-goals

- Do not query live mixers for these surfaces in this slice.
- Do not treat cached DTO fixture evidence as physical hardware proof.
- Do not add live read/write buttons or promote Full Autonomous mode.

## Safety constraints

- Gap IDs are evidence labels only.
- The proof evidence row remains `canWriteLive=false`.
- Full Autonomous readiness remains blocked while physical readback gaps exist.

## Expected evidence

- Frontend adapter keeps gap surface IDs in the supervised proof evidence row.
- Settings inspector shows a compact gap surface summary.
- E2E backend DTO coverage proves the browser-visible summary.
