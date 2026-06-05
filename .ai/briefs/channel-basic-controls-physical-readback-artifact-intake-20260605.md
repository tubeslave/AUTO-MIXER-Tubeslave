# Channel Basic Controls Physical Readback Artifact Intake

## Context

Channel identity/basic-control readback has a schema contract for future physical WING evidence, but the repo still needs a safe place to inventory operator-captured artifacts before any fixture can be reviewed.

## Goal

Add a read-only intake for `channel_identity_and_basic_controls` physical readback artifacts. The intake may identify valid candidates for manual review, but it must not query hardware, mutate console state, close fixture gaps, or grant live authority.

## Safety Constraints

- Keep `live_write=false`, `mutates_console_state=false`, `queries_console=false`, and `executes_tests=false`.
- Keep `fixture_closure_allowed_by_intake=false`.
- Keep `physical_fixture_gap_remains=true`.
- Keep `live_authority_granted=false` and `full_autonomous_ready=false`.
- Expose any candidate through no-WING/Product Layer/frontend as read-only evidence only.

## Test Plan

- Unit test missing, valid, invalid, discovery, registry, and capability matrix behavior.
- Validate no-WING schema, summary embedding, and run order.
- Validate Product Layer normalization of unsafe source claims.
- Validate frontend adapter contract rows and fallback behavior.
