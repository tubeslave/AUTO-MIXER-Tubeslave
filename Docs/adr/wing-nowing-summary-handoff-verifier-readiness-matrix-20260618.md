# WING no-WING Summary Verifier Readiness Matrix Boundary

## Context

The handoff packet now includes
`artifact_manifest_verifier_readiness_matrix_update_allowed: false`, but the
no-WING release summary embeds reduced plan and handoff packet artifact
summaries. The embedded summaries are the surfaces operators and automated gates
inspect before a real WING read-only session.

## Options considered

- Leave the full handoff packet as the only source for this verifier
  readiness-matrix boundary.
- Mirror the field into the no-WING embedded plan and handoff summary JSON and
  Markdown.

## Decision

Mirror `artifact_manifest_verifier_readiness_matrix_update_allowed: false` into
the no-WING embedded plan artifact summary, handoff packet artifact summary, and
rendered Markdown.

## Why this won

The current WING command requires the no-WING handoff summary to expose the
verifier no-authority boundary. Readiness matrix updates are a separate
authority surface, so both preflight summary artifacts must show the same denial
as the full packet and the verifier report.

## Rejected alternatives

Relying on the full handoff packet alone was rejected because summary consumers
could miss the readiness-matrix denial marker when inspecting the embedded plan
or handoff artifact surface.

## Implementation plan

- Copy the verifier readiness-matrix denial from the prestart verifier report
  into the offline preflight plan.
- Validate the field in the no-WING plan artifact schema.
- Add the field to `wing_readonly_preflight_plan_artifact_summary`.
- Add the field to `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render the field in no-WING `summary.md`.
- Add JSON and Markdown assertions in `tests/test_no_wing_release_check.py`.

## Test plan

- Targeted no-WING embedded handoff summary test.
- No-WING quick release check.
- Focused WING/no-WING safety suite.
- Full pytest before commit.

## Risks and rollback

The change is additive and read-only. Roll back by removing the embedded field,
Markdown line, tests, and this ADR if a newer summary contract replaces the
handoff verifier readiness-matrix boundary.
