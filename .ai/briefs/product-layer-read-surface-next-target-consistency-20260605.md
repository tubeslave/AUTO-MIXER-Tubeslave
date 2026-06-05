# product-layer-read-surface-next-target-consistency-20260605

## Context

no-WING read-surface summaries now expose `next_physical_readback_surface_id`.
Product Layer still derives the target from the physical gap list without
surfacing whether the source field matches that list.

## Goal

Normalize the source next-readback target in Product Layer and expose a
fail-closed consistency flag through Settings.

## Non-goals

- Do not query WING or dLive hardware.
- Do not execute the next proof target.
- Do not promote Full Autonomous mode.

## Safety constraints

- Source mismatch must remain read-only evidence and should not grant authority.
- Product Layer may derive a display fallback from the first physical gap, but
  it must surface the mismatch.
- Settings must show the consistency state separately from the target label.

## Expected evidence

- Backend proof-quality payload includes source target and consistency flag.
- Frontend adapter preserves the consistency flag.
- Settings inspector renders the read-target gate.
- Tests cover consistent and mismatched source artifacts.
