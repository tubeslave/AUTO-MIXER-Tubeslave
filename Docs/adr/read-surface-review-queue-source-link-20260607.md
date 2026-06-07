# ADR: Read-Surface Review Queue Source Link

## Context

The read-surface artifact review queue aggregates per-surface physical readback
intake snapshots. Downstream acceptance templates need an `artifact_source_id`
to create operator review records. Real intake builders expose valid candidates
through `latest_valid_candidate_source_id` and valid artifact records, while the
queue previously looked only for `artifact_source_ids`.

## Options considered

1. Require each intake builder to add `artifact_source_ids`.
2. Let the acceptance template inspect intake snapshots directly.
3. Normalize candidate source ids in the aggregate review queue.

## Decision

Use option 3. The review queue now derives `artifact_source_ids` from explicit
summary ids, `latest_valid_candidate_source_id`, and accepted artifact records.

## Why this won

The review queue is the contract boundary between intake validation and operator
acceptance. Normalizing source ids there keeps downstream templates simple and
prevents untraceable candidates from being manually accepted.

## Rejected alternatives

- Updating every intake builder is broader than needed and risks churn across
  ten surfaces.
- Having the acceptance template inspect intake snapshots would bypass the queue
  as the operator handoff.
- Allowing source-less candidates creates review records that cannot point back
  to a physical readback artifact.

## Implementation plan

1. Normalize source ids from summary and artifact records in the queue item.
2. Require a source id before surfacing a valid item as a review candidate.
3. Report `candidate_source_id_missing` at item and summary level.
4. Extend no-WING validation and Markdown summaries for the new fail-closed
   condition.
5. Cover real raw-mixer intake and missing-source-id cases in tests.

## Test plan

- `python3 -m py_compile backend/read_surface_physical_readback_artifact_review_queue.py tools/build_read_surface_physical_readback_artifact_review_queue.py tools/no_wing_release_check.py`
- Focused review-queue, acceptance-template, and no-WING pytest.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest.

## Risks and rollback

This change does not query hardware, mutate a console, update fixtures, close
read-surface gaps, or grant live authority. Rollback is to remove the queue
source-id normalization, no-WING summary fields, tests, and docs row; no mixer
transport path is affected.
