# WING MVP Supervised Prestart Marker Count Summary Fields

## Summary

The no-WING handoff summary now exposes the supervised prestart gate required
marker count, expected marker count, and two count-match verdicts proving the
structured gate count matches both the shared contract and the top-level summary
marker list.

## Fields

- `supervised_prestart_gate_required_marker_count`
- `expected_supervised_prestart_gate_required_marker_count`
- `supervised_prestart_gate_required_marker_count_matches_contract`
- `supervised_prestart_gate_required_marker_count_matches_summary`

## Safety Boundary

- No WING hardware is contacted.
- No live write authority is granted.
- No readiness closure is allowed by these fields.
- The WING read-only connection proof blocker remains open until a real
  `wing_readonly_preflight.v1` artifact is captured and manually reviewed.

## Validation

- Python compile for touched files passed.
- Focused WING/no-WING pytest passed: `386 passed, 8 skipped`.
- no-WING quick release check passed with marker count `5`, expected count `5`,
  and both count-match verdicts `true`.
- Project Director stayed on `WING read-only connection proof` and included all
  marker count fields in the selected action.
- Full pytest suite passed: `3446 passed, 9 skipped`.
- Memory lint passed.
- Git diff whitespace check passed.
