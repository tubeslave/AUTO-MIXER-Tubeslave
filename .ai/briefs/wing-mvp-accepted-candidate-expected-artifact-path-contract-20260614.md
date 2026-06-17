# Brief: WING MVP Accepted Candidate Expected Artifact Path Contract

## Task

Close a fail-open gap in the accepted-candidate prestart contract used before
supervised WING proof planning.

## Context

The review template now rejects unbound expected artifact paths, but the shared
accepted-candidate prestart helper still trusted non-empty expected artifact
fields plus boolean match flags. A stale or hand-built candidate row could keep
`summary_path_matches_expected_artifact: true` while pointing
`expected_summary_artifact` or `expected_events_artifact` at a different path.

## Constraints

- Stay on the first WING MVP blocker.
- Keep all work read-only/offline.
- Do not grant live authority or close readiness from candidate data.
- Preserve existing no-WING and supervised proof-plan gates.

## Desired Outcome

The shared contract recomputes expected-artifact path identity and no-WING
validation fails closed when an accepted candidate carries mismatched expected
artifact paths before physical proof steps.
