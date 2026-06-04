# Proposal: snapshots-scenes-read-dto-20260604

## Short thesis

Add a cached-only Product Layer snapshots/scenes catalog DTO that makes scene
state visible without making scene load/save executable.

## What I understood

The operator cockpit needs visibility into show/snapshot names and active-scene
markers, but snapshots/scenes are too destructive for automation. The safe next
step is display/read coverage, not queueing or applying scene changes.

## Proposed solution

Implement `build_snapshots_scenes_snapshot(server)` in
`backend/operator_product_state.py`. It reads only `mixer_client.state`, accepts
several cache shapes, publishes per-slot read coverage, and labels scene load
and save as blocked manual-only policies.

Register `get_snapshots_scenes_snapshot` in the read-only product-state handler.
Update the read surface inventory so snapshots/scenes move from raw query
language to Product snapshot coverage while staying `partial`.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `tests/test_read_surface_inventory.py`

## Alternatives considered

- Call `find_snap_by_name()` or scan snapshot names on demand. Rejected because
  it sends OSC and can create false confidence about scene safety.
- Add a scene proposal queue item. Rejected because scene diff, rollback, and
  live proof are not present.

## Risks

- Cached names may be stale or incomplete.
- Flat WING state cannot represent a full indexed scene catalog unless another
  safe reader has populated a structured cache.

## Test plan

- Unit test cached catalog DTO with active scene metadata.
- Unit test missing cache behavior.
- Handler round-trip test for the new message type.
- Inventory test for `partial` status and destructive marker fields.
- Full suite before reporting completion.

## Where the other agent may disagree

Kimi may prefer leaving snapshots/scenes only in the declarative proposal layer
until physical query fixtures exist. My counterpoint is that a cached-only DTO
improves operator visibility without changing the apply boundary.
