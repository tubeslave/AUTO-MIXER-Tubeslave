# Brief: fx-parameter-schema-hints-20260605

## Context

The project goal requires autonomous FX management and FX send tuning, but the
live write boundary still blocks FX and insert apply paths. The Product Layer
already exposes cached FX slot metadata, numeric parameter slots, insert state,
and aggregate cache freshness without querying or mutating the console.

The next gap is interpretation: `/fx/<slot>/<parameter>` values are visible as
raw numeric slots, so downstream assisted proposals cannot safely explain what a
parameter means.

## Task

Expose read-only FX parameter schema hints for known generic FX families:
reverb, delay, and modulation. Unknown FX models must remain visible as generic
numeric parameters instead of being mislabelled.

## Constraints

- Do not query hardware.
- Do not mutate mixer state.
- Do not add FX or insert apply authority.
- Do not claim physical WING model schema verification.
- Keep complete model-specific FX schemas, physical fixtures, scoped insert
  diffs, and FX safety review as blockers.

## Acceptance

- `fx_inserts_snapshot` includes per-parameter semantic hints and slot-level
  schema summary.
- Unknown FX models fall back to `generic_numeric`.
- Inventory and fixture evidence distinguish schema hints from physical proof.
- Focused Product Layer/read-surface tests and full test suite pass.
