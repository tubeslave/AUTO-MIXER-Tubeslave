# ADR: WING MVP Current Command Director Digest Rendered Fields

## Context

The WING no-WING handoff summary renders both declared and recomputed Project
Director selected-action digest fields. `Current First Command` had started to
name those fields, but Project Director baseline coverage only required the
shorter digest subset. That made the action contract weaker than the rendered
handoff proof surface.

## Options considered

1. Keep the shorter baseline marker set.
2. Add only expected/actual digest field names to the current command text.
3. Add the rendered expected/actual digest fields and coverage marker-count
   field to baseline markers, autoloop docs, and rails tests.

## Decision

Use option 3.

## Why this won

The selected action is the instruction that autonomous Goal-mode cycles follow.
If no-WING handoff output must render expected and actual Project Director
digest fields, the Project Director coverage gate should also require those
fields. This keeps the offline handoff rail aligned without granting live
authority.

## Rejected alternatives

The shorter marker set was too permissive. Text-only updates without baseline
markers would document the fields but would not fail when future edits remove
them.

## Implementation plan

Extend `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS` with
`expected_project_director_selected_action_anchor`,
`expected_project_director_selected_action_source: Current First Command`,
`actual_project_director_selected_action_sha256`,
`expected_project_director_selected_action_sha256`,
`project_director_selected_action_sha256_available_matches_digest: true`, and
`project_director_current_first_command_baseline_coverage_marker_count`. Add
the same requirements to current command/autoloop docs and regression tests.

## Test plan

Run focused Project Director, WING rail, autoloop prompt, and autoloop contract
tests. Then run Project Director, direction guard, no-WING quick fail-closed,
memory lint, `git diff --check`, and the standard full test command.

## Risks and rollback

Risk is limited to offline planning text and baseline gating. Rollback is to
remove the added markers and assertions if the no-WING handoff summary stops
rendering these fields. This change does not start WING, query a console, write
to a console, update readiness, or close the blocker.
