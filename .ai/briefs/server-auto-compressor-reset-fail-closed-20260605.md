# Server Auto Compressor Reset Fail Closed - 2026-06-05

## Task

Close the remaining server-side Auto Compressor reset helper so it returns blocked evidence instead of calling mixer-client compressor write methods.

## Context

- The approved live-write path is still limited to supervised Product Layer `fader` and `gain` writes.
- AutoCompressor and CF compressor controller apply paths already return blocked evidence.
- `AutoMixerServer._reset_compressor_on_channel` still contains direct `set_compressor_on` and `set_compressor` calls, even though WING deployment entrypoints are quarantined before reaching it.

## Constraints

- Do not widen live writes.
- Do not call OSC or mixer write methods.
- Preserve the would-be reset parameters as structured blocked evidence.
- Keep the change focused on the server helper and its safety metadata.

## Expected Output

- `_reset_compressor_on_channel` returns fail-closed evidence with `mutates_mixer=False`.
- Tests prove the helper does not call `set_compressor_on` or `set_compressor`.
- Inventory, capability matrix, processing inventory, and ADR mention the server reset helper.
