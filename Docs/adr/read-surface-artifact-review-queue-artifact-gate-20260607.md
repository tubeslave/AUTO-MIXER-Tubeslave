# ADR: Read-Surface Artifact Review Queue Artifact Gate

## Context

The read-surface physical-readback artifact review queue consolidates intake
results before operators create acceptance records. The existing no-WING release
check validated the in-memory queue, but it did not generate and validate a
queue artifact between capture-packet generation and acceptance-template
generation.

## Options considered

1. Keep the in-memory schema check only.
2. Let acceptance-template generation imply a valid review queue.
3. Generate a review-queue artifact and validate it inside no-WING.

## Decision

Use option 3. no-WING now writes
`read_surface_physical_readback_artifact_review_queue.json` and validates the
generated artifact immediately after the builder command succeeds.

## Why this won

The review queue is the boundary between raw physical-readback artifacts and
operator acceptance records. A generated artifact gate proves that release runs
carry forward a safe queue state: review may be required, but fixture closure,
live widening, live authority, and full autonomous readiness remain false.

## Rejected alternatives

- In-memory validation alone does not prove the release run produced the
  operator handoff artifact.
- Acceptance-template generation should not silently stand in for the queue
  review boundary.
- Automatic acceptance would skip the explicit human review this phase requires.

## Implementation plan

1. Add an offline review-queue builder with Markdown and JSON output.
2. Add the builder to no-WING release checks after the capture packet.
3. Validate the generated artifact for read-only safety and count consistency.
4. Add JSON/Markdown summary output for the artifact gate.
5. Cover valid, missing, unsafe, ordering, CLI, and summary cases in tests.

## Test plan

- `python3 -m py_compile backend/read_surface_physical_readback_artifact_review_queue.py tools/no_wing_release_check.py tools/build_read_surface_physical_readback_capture_packet.py tools/build_read_surface_physical_readback_artifact_review_queue.py tools/build_read_surface_physical_readback_fixture_closure_plan.py`
- Focused no-WING pytest for review-queue artifact validation.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest.

## Risks and rollback

This gate does not accept candidates, update fixtures, or prove live readiness.
Operators still need real physical artifacts, review decisions, acceptance
records, and a later manual fixture-baseline update. Rollback is to remove the
builder, validator, tests, and docs; no mixer transport path is affected.
