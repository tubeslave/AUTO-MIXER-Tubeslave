# wing-ratio-signal-analysis-canonical-table-20260612

Status: superseded for live OSC encoding on 2026-07-12. The table below remains
a legacy signal-analysis display quantizer; live writes use the fw3.0.5 string
enum documented in `Docs/TOOLS.md` and implemented by
`backend/compressor_adaptation.py`.

## Context

`signal_analysis.ratio_float_to_wing()` is a helper used around compressor signal
analysis. Its local ratio table had drifted from the WING protocol table recorded
in `Docs/TOOLS.md`, which can make analysis output disagree with the console
ratio indices expected by downstream automation.

## Options considered

1. Leave the current helper table as-is because the focused tests passed.
2. Remove the helper and force all callers to depend on another compressor module.
3. Align the helper's table and tests with the documented WING protocol order.

## Decision

Use option 3. Keep the helper local to `signal_analysis.py`, but make its
`WING_RATIO_VALUES` and `WING_RATIO_STRINGS` match `Docs/TOOLS.md`.

## Why this won

It is the smallest change that restores protocol parity without introducing a
new dependency, changing live-write authority, or widening the supervised
execution path.

## Rejected alternatives

Keeping the old table was rejected because passing tests only proved internal
consistency, not console protocol correctness. Removing the helper was rejected
because it would be a broader refactor for a narrow contract fix.

## Implementation plan

1. Replace the local ratio table with the documented WING ratio values and labels.
2. Clamp unsupported infinite ratios to the maximum documented WING ratio.
3. Extend tests to assert the full protocol table order.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_signal_analysis.py tests/test_compressor.py::TestRatioConversion tests/test_gcc_phat.py`

## Risks and rollback

The helper now returns labels such as `2:1` instead of `2.0:1`. If a downstream
consumer requires the previous display format, add an explicit display adapter
instead of changing the protocol table. Rollback is reverting this ADR, the
`signal_analysis.py` table, and the focused tests.
