# Proposal: user-presets-read-dto-20260604

## Short thesis

Add a cached-only Product Layer DTO for user presets so the cockpit can display
known preset slots without creating a preset recall/store path.

## What I understood

The project goal includes user preset control, but the current safe milestone is
read coverage. Preset recall/store remains destructive and must stay blocked
until readback, scoped diff, and rollback proof exist.

## Proposed solution

Implement `build_user_presets_snapshot(server)` in
`backend/operator_product_state.py`. It reads only `mixer_client.state`, accepts
structured cached catalogs, publishes per-slot read coverage, and marks preset
recall/store as blocked manual-only policies.

Register `get_user_presets_snapshot` in product-state handlers and update the
read surface inventory from `missing` to `partial`.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`

## Alternatives considered

- Leave user presets as missing. Rejected because a cached Product DTO moves the
  read coverage map forward without changing apply authority.
- Add preset recall/store proposals. Rejected because preset contents, scoped
  diff, and rollback proof are not present.

## Risks

- Cached preset metadata may be stale or incomplete.
- A preset name alone does not prove safe recall semantics.

## Test plan

- Unit test populated cached catalog without transport calls.
- Unit test missing cache behavior.
- Handler round trip for the new DTO.
- Inventory test that user presets remain `partial`.
- Full suite before reporting completion.

## Where the other agent may disagree

Kimi may prefer requiring physical preset enumeration first. My counterpoint is
that this DTO is explicitly cached-only and improves operator visibility while
keeping recall/store blocked.
