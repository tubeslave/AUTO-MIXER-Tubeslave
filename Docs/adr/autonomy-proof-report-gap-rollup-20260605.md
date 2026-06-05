# ADR: autonomy-proof-report-gap-rollup-20260605

## Context

`tools/evaluate_supervised_proof_artifacts.py` builds the local proof report
used before widening any supervised or autonomous authority. The report already
includes `autonomy_mode_transition_gate`, but the top-level summary does not
show the Full Autonomous readiness gap buckets.

## Options considered

1. Add a top-level report rollup sourced from the mode gate.
2. Keep the data only in nested mode gate rows.
3. Create a new remediation report module.

## Decision

Use option 1. Add `summary.autonomy_gap_rollup` to the proof artifact report and
add matching aggregate fields to `autonomy_mode_transition_gate.summary`.
The no-WING release runner should validate the generated report JSON after the
proof-report command completes.
The expected Supervised live-write kinds come from `operator_mode_policy`, not
from a duplicate constant in the release runner.

## Why this won

The proof report becomes more useful as a readiness artifact while keeping one
canonical source of truth: the rollup is derived from the mode gate, not from a
separate parser.

## Rejected alternatives

Nested-only data is easy to miss in operator review. A new remediation module
adds more surface area before there is a distinct workflow to justify it.

## Implementation plan

1. Summarize Full Auto readiness gap counts and target IDs in the mode gate.
2. Summarize Supervised current live-write kinds in the mode gate.
3. Add `build_autonomy_gap_rollup()` to the proof report tool.
4. Attach the rollup to `report["summary"]` after the mode gate is built.
5. Add a local artifact schema check to `tools/no_wing_release_check.py`.
6. Derive the expected Supervised live-write kinds from `operator_mode_policy`.
7. Include the rollup in `no_wing_release_check` JSON and Markdown summaries.
8. Extend focused tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py tests/test_supervised_proof_artifact_report.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `python3 -m py_compile backend/autonomy_mode_transition_gate.py tools/evaluate_supervised_proof_artifacts.py`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-current --command-timeout-sec 300`

## Risks and rollback

Risk: the rollup may be mistaken for authorization. The safety fields and claim
explicitly say it is report-only, does not widen live authority, and keeps Full
Autonomous blocked.

Rollback is removing the summary fields, `build_autonomy_gap_rollup()`, the test
assertions, the release-check schema validator, the no-WING summary section, and
this ADR.
