# Proposal: channel-processing-field-freshness-20260605

## Short thesis

Expose per-field freshness for cached channel-processing state before any
autonomous EQ, compressor, or gate promotion work.

## What I understood

The project goal requires full console understanding, but the current approved
live path remains limited to supervised `fader` and `gain`. For processing
surfaces, the next safe step is better observation: the product snapshot should
show whether HPF, EQ bands, compressor parameters, and gate timing values are
fresh, stale, unknown, or missing.

## Proposed solution

Add a small helper in `operator_product_state` that maps read fields to one or
more cached console addresses and returns freshness metadata for each field. Use
that helper only inside `build_channel_processing_snapshot` and its nested HPF,
EQ, compressor, gate, and EQ-band DTOs. Keep top-level aggregate freshness
unchanged, but add section-local freshness so later proposal builders can reject
stale processing evidence precisely.

## Likely files to touch

- `backend/operator_product_state.py`
- `backend/read_surface_inventory.py`
- `tests/test_operator_product_state.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`

## Alternatives considered

- Add physical OSC query fixtures first. Better long term, but this cycle can
  safely improve the cached DTO without contacting hardware.
- Add freshness only to the top-level snapshot. Rejected because autonomous
  processing decisions need to know which field or section is stale.
- Promote EQ/compressor/gate proposals to queueable. Rejected because readback,
  rollback, and physical proof are still incomplete.

## Risks

- Payload size grows because each processing section now carries per-field
  metadata.
- Consumers that assume `read_coverage` contains only coverage fields must
  tolerate the new nested `cache_freshness` key.

## Test plan

- Focused product-state tests for fresh and stale channel-processing cache.
- Read-surface inventory and remediation tests.
- Compile touched modules.
- Full suite and `git diff --check`.

## Where the other agent may disagree

Kimi may argue that physical query fixtures should come before DTO enrichment. I
agree they are required before promotion, but field-level cached freshness is a
low-risk prerequisite and makes the physical fixture gap more explicit.
