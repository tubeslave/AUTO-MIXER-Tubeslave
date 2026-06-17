# Proposal: WING MVP Read-Only Event Log Run-Id Binding

## 1. Short thesis

Add an explicit run-id binding between real WING preflight `summary.json` and
its `events.jsonl` rows.

## 2. What I understood

The active MVP remains supervised WING Rack proof. We cannot close it without a
real WING read-only capture, but we can make that future capture harder to
mis-review.

## 3. Proposed solution

Have `tools/wing_readonly_preflight_runner.py` write the current run id into
each event log row. Then make `backend/wing_readonly_preflight_evidence.py`
reject event logs whose row-level `run_id` does not match the summary run id.
Expose the verdict in the review report.

## 4. Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`

## 5. Alternatives considered

- Rely only on `events_path` and SHA256. Rejected because that does not show
  row-level identity inside the log.
- Add a separate manifest file for real WING preflight runs. Deferred as a
  larger packaging change.

## 6. Risks

Older captured artifacts without row-level run ids will become invalid under the
new validator. That is acceptable for this MVP gate because the first blocker
requires a fresh real WING capture.

## 7. Test plan

Run focused runner/evidence/review tests, WING rail tests, the full standard
pytest command, no-WING runtime, `git diff --check`, memory lint, and Project
Director.

## 8. Where the other agent may disagree

Kimi may prefer a full artifact manifest for the real preflight run. This
targeted binding is a smaller step that directly closes the current identity
gap.
