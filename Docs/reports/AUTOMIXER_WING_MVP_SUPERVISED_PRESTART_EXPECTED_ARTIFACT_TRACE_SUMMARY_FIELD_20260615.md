# WING MVP Supervised Prestart Expected Artifact Trace Summary Field

## Summary

The no-WING handoff summary now exposes
`supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace`
as a top-level JSON and Markdown field derived from the structured
`supervised_prestart_gate`.

## Why

The supervised fader/gain proof must remain blocked until an accepted read-only
candidate has an expected-artifact trace. A flat summary field lets operators
and automation verify that requirement without parsing the nested gate object.

## Safety Boundary

- No WING hardware is contacted.
- No live write authority is granted.
- No readiness closure is allowed by this field.
- The current WING read-only connection proof blocker remains open until a real
  `wing_readonly_preflight.v1` artifact is captured and manually reviewed.

## Validation

- Python compile for touched files passed.
- Focused pytest passed: `386 passed, 8 skipped`.
- no-WING quick release check passed with
  `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace:
  true`.
- Project Director stayed on `WING read-only connection proof` with baseline
  marker count `197` and missing markers `[]`.
- Full pytest suite passed: `3446 passed, 9 skipped`.
- Memory lint passed.
- Git diff whitespace check passed.
