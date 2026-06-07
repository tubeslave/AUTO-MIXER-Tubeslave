# Proposal: Read-Surface Review Queue Source Link

## 1. Short thesis

Carry source ids through the read-surface review queue so valid physical
readback candidates can be reviewed by operators without losing their artifact
identity.

## 2. What I understood

The project is building a safe path from physical readback capture to fixture
closure. The review queue is the aggregate handoff, but downstream acceptance
records require `artifact_source_id`. Real intake builders provide
`latest_valid_candidate_source_id`; the queue should normalize that into
`artifact_source_ids`.

## 3. Proposed solution

- Add source-id normalization in
  `backend/read_surface_physical_readback_artifact_review_queue.py`.
- Treat a valid claimed candidate with no source id as
  `operator_artifact_replacement_required`.
- Add item and summary fields for `candidate_source_id_missing`.
- Extend no-WING snapshot/artifact validation and summary output.
- Add tests that prove a real raw-mixer intake with `inline_payload_0` reaches
  the acceptance-template record.

## 4. Likely files to touch

- `backend/read_surface_physical_readback_artifact_review_queue.py`
- `tools/build_read_surface_physical_readback_artifact_review_queue.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_artifact_review_queue.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_template.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-review-queue-source-link-20260607.md`
- `Docs/adr/read-surface-review-queue-source-link-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_REVIEW_QUEUE_SOURCE_LINK_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Leave source ids as a field only synthetic intake summaries may provide.
- Let acceptance-template generation inspect raw intake artifacts directly.
- Allow review candidates without source ids.

## 6. Risks

The main risk is accepting an untraceable candidate. The queue should fail
closed when a source id is missing, and no-WING should report the missing link.

## 7. Test plan

- py_compile for the backend queue, queue CLI, and no-WING runner.
- Focused review-queue, acceptance-template, and no-WING tests.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest because no-WING summary behavior changes.

## 8. Where the other agent may disagree

Kimi may prefer changing every intake builder to emit `artifact_source_ids`
directly. The narrower queue-level normalization is safer here because it
preserves existing intake contracts while fixing the aggregate handoff.
