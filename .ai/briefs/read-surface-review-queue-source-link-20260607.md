# Brief: Read-Surface Review Queue Source Link

## Task

Preserve physical readback artifact source ids when valid intake candidates flow
into the aggregate review queue.

## Context

The per-surface intake builders expose valid operator-captured artifacts through
`latest_valid_candidate_source_id` and artifact records. The aggregate review
queue previously only used `artifact_source_ids` when that field was already
present in the summary. That made synthetic tests pass, but a real valid intake
could become a review candidate without a source id for the downstream
acceptance-template record.

## Safety Boundary

- Do not query or mutate a mixer.
- Do not accept physical readback into fixture baselines.
- Do not close fixture gaps or grant live authority.
- If a valid candidate cannot be linked to a source id, require artifact
  replacement rather than manual acceptance.

## Acceptance Criteria

- Review queue items normalize source ids from `artifact_source_ids`,
  `latest_valid_candidate_source_id`, and valid artifact records.
- Valid candidates without any source id are not surfaced as fixture candidates.
- no-WING release checks report and reject candidate-source-id gaps.
- Acceptance-template generation can create a pending review record from a real
  intake candidate that only exposes `latest_valid_candidate_source_id`.
