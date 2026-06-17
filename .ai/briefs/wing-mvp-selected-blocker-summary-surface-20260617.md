# Brief: WING MVP Selected Blocker Summary Surface

## Task ID

wing-mvp-selected-blocker-summary-surface-20260617

## Goal

Expose the selected WING MVP blocker and blocker status at the no-WING
`summary.json` top level when a WING read-only preflight handoff packet is
present.

## Context

The current Project Director guard still selects `WING read-only connection
proof` as the first open MVP blocker. The no-WING handoff packet already carries
`selected_mvp_blocker` and `selected_mvp_blocker_status`, but the top-level
summary requires readers to inspect the nested packet artifact to confirm the
packet remains bound to the read-only WING blocker.

## Scope

- Mirror the selected blocker and blocker status from the handoff packet into
  the existing MVP blocker summary surface.
- Keep all live authority and readiness closure fields false.
- Do not contact WING hardware or broaden live write authority.
- Do not touch unrelated dirty memory-trust or queue-import worktree drift.

## Validation

- Compile the touched no-WING check and test file.
- Run focused no-WING/WING guard tests.
- Run no-WING quick and verify the new top-level fields in a fresh artifact.
- Run full pytest before commit.
