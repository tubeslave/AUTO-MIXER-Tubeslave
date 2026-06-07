# Proposal: Read-Surface Artifact Review Queue Artifact Gate

## 1. Short thesis

Make the physical-readback artifact review queue a generated, validated no-WING
artifact before acceptance-template generation.

## 2. What I understood

The project is still proving read surfaces without live console authority. The
capture packet tells an operator what artifact to capture, while the acceptance
template assumes a queue of review candidates exists. That queue should be a
file-backed release artifact, not only an in-memory check.

## 3. Proposed solution

- Add `tools/build_read_surface_physical_readback_artifact_review_queue.py`.
- Generate Markdown for operators and JSON for no-WING release checks.
- Validate the generated JSON artifact for safety flags, count consistency,
  candidate state, blocked reasons, operator sequence, and focused validation.
- Insert the builder and validator after capture-packet generation and before
  acceptance-template generation.
- Add the artifact result to `summary.json` and `summary.md`.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_artifact_review_queue.py`
- `backend/read_surface_physical_readback_artifact_review_queue.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_artifact_review_queue.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-artifact-review-queue-artifact-gate-20260607.md`
- `Docs/adr/read-surface-artifact-review-queue-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_ARTIFACT_REVIEW_QUEUE_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Keep the review queue as an in-memory schema check only.
- Let the acceptance template imply the queue state.
- Automatically accept review candidates when their intakes are valid.

## 6. Risks

The main risk is mistaking a queue artifact for operator acceptance. The artifact
must keep fixture closure, live widening, live authority, and autonomous
readiness false even when review candidates exist.

## 7. Test plan

- py_compile for the no-WING tool and builders.
- Focused tests for the builder and valid/missing/unsafe artifacts.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may argue that another artifact gate slows the path to hardware. The reason
to keep it is that acceptance and fixture closure depend on a trustworthy queue:
the generated review boundary should be explicit, file-backed, and fail-closed.
