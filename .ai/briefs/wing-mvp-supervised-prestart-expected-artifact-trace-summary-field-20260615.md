# Brief: WING MVP Supervised Prestart Expected Artifact Trace Summary Field

## Task

Expose the supervised prestart expected-artifact trace gate as a top-level
no-WING summary field.

## Context

The handoff packet already carries a structured `supervised_prestart_gate` with
`requires_accepted_candidate_expected_artifact_trace: true`. Operators and
automation still need a grep-friendly top-level `summary.json` and
`summary.md` field before any supervised fader/gain proof planning.

## Scope

- Add a derived no-WING handoff summary field:
  `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace`.
- Mirror the field into the top-level no-WING JSON surface and Markdown summary.
- Require the marker in Project Director Current First Command baseline
  coverage.
- Keep all behavior offline and read-only.

## Out Of Scope

- Running WING hardware.
- Granting live authority.
- Closing the WING read-only connection proof blocker.
