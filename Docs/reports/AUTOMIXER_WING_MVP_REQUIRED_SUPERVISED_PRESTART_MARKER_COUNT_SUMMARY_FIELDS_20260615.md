# WING MVP Required Supervised Prestart Marker Count Summary Fields

## Summary

The no-WING handoff summary now exposes the top-level required supervised
prestart marker count, expected count, and two count-match verdicts proving the
top-level marker count matches both the shared contract and the structured
supervised gate marker count.

## Fields

- `required_supervised_prestart_gate_marker_count`
- `expected_required_supervised_prestart_gate_marker_count`
- `required_supervised_prestart_gate_marker_count_matches_contract`
- `required_supervised_prestart_gate_marker_count_matches_supervised_gate`

## Safety Boundary

- No WING hardware is contacted.
- No live write authority is granted.
- No readiness closure is allowed by these fields.
- The WING read-only connection proof blocker remains open until a real
  `wing_readonly_preflight.v1` artifact is captured and manually reviewed.

## Validation

- Python compile for touched files passed.
- Focused WING/no-WING pytest passed: `386 passed, 8 skipped`.
- no-WING quick release check passed with required marker count `5`, expected
  count `5`, and both new count-match verdicts `true`.
- Project Director stayed on `WING read-only connection proof`, included all new
  marker count fields in the selected action, and reported baseline marker count
  `218` with missing markers `[]`.
- Full pytest suite passed: `3446 passed, 9 skipped`.
- Memory lint passed.
- Git diff whitespace check passed.
