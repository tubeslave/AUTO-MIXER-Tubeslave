# Codex Proposal: Matrix Sends Read DTO - 2026-06-05

## Short thesis

Add cached channel-to-matrix send visibility to the Product Layer so routing and matrix decisions can inspect matrix feeds without gaining live write authority.

## What I understood

The target AI Mixing Engineer must analyze routing and matrices before it can safely build or adapt a live mix. The current routing/sends DTO covers channel-to-bus sends, channel-to-main sends, output route sources, and static Dante roles. Matrix sends remain a named blocker even though their cached WING addresses can be normalized read-only.

## Proposed solution

- Add `_channel_matrix_send_row` beside the existing bus/main send row builders.
- Read cached `/ch/N/send/MXK/*` addresses only.
- Add `matrix_send_count`, `active_matrix_send_count`, and `channel_matrix_sends` to `build_routing_sends_snapshot`.
- Update read-surface inventory, fixture evidence, and focused tests.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `backend/read_surface_fixture_evidence.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`
- `tests/test_read_surface_fixture_evidence.py`
- `Docs/adr/matrix-sends-read-dto-20260605.md`

## Alternatives considered

- Query matrix sends from WING. Rejected because Product Layer snapshots must remain cached/read-only.
- Put matrix sends directly inside `matrix_snapshot`. Rejected for now because the known addresses are channel send addresses and match the existing routing/sends surface.
- Promote matrix send apply. Rejected because routing writes remain outside the proven supervised executor boundary.

## Risks

- Consumers may overread cached matrix-send state as physical route proof. The snapshot remains partial autonomous evidence with coverage flags and safety metadata.

## Test plan

- Add focused routing/sends snapshot tests for cached and missing matrix-send rows.
- Run operator product-state, read-surface inventory, fixture evidence, registry/capability tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may prefer a broader route graph abstraction before adding another row family. That abstraction is likely useful later, but this patch closes the current blocker with the existing DTO pattern and no live-write expansion.
