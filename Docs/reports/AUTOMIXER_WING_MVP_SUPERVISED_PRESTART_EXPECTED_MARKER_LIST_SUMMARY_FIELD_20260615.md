# WING MVP Supervised Prestart Expected Marker List Summary Field

## Summary

The no-WING handoff summary now exposes
`expected_supervised_prestart_gate_required_markers` as a flat field next to the
actual `supervised_prestart_gate_required_markers` list, marker counts, and
match verdicts.

## Field

- `expected_supervised_prestart_gate_required_markers`

## Safety Boundary

- No WING hardware is contacted.
- No live write authority is granted.
- No readiness closure is allowed by this field.
- The WING read-only connection proof blocker remains open until a real
  `wing_readonly_preflight.v1` artifact is captured and manually reviewed.

## Validation

- Python compile for touched files passed.
- Focused WING/no-WING pytest passed: `386 passed, 8 skipped`.
- no-WING quick release check passed with
  `expected_supervised_prestart_gate_required_markers` rendered beside the
  actual marker list.
- Project Director stayed on `WING read-only connection proof`, included the
  expected marker list in the selected action, and reported baseline marker count
  `222` with missing markers `[]`.
- Full pytest suite passed: `3446 passed, 9 skipped`.
- Memory lint passed.
- Git diff whitespace check passed.
