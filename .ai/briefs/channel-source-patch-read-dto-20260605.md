# Channel Source Patch Read DTO - 2026-06-05

## Task

Expose cached channel input source patch state in the Product Layer channel inventory without querying or mutating a mixer.

## Context

- The project goal requires reading all available console parameters before broad autonomous control.
- `build_channel_inventory` currently exposes fallback source labels like `In 02`, but not cached WING input patch fields.
- WING source patch writes (`set_channel_input`, `set_channel_alt_input`) are destructive routing changes and remain quarantined.
- Cached addresses `/ch/N/in/conn/grp`, `/ch/N/in/conn/in`, `/ch/N/in/conn/altgrp`, and `/ch/N/in/conn/altin` are enough to publish a read-only DTO when present.

## Constraints

- Do not call `send`, `get_channel_input_routing`, or any routing setter.
- Do not add queueable routing proposals or live apply fields.
- Preserve backward-compatible channel fields while adding explicit source-patch coverage and cache freshness.
- Keep the read-surface inventory honest: this is cached Product Layer evidence, not physical proof.

## Expected Output

- Channel inventory rows include `source_patch` with main/alt source group/channel, display labels, read coverage, and field freshness.
- The aggregate channel read coverage tracks the new `source_patch` section.
- Read-surface inventory and fixture evidence mention cached source-patch coverage while retaining physical-readback blockers.
- Tests prove no mixer query or write method is called.
