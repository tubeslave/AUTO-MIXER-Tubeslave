# ADR: WING Read-Only Preflight Review Release Gate

## Context

The WING read-only preflight path now has two separate offline checks:

- `backend/wing_readonly_preflight_evidence.py` validates captured
  `wing_readonly_preflight.v1` summaries and matching `events.jsonl` logs.
- `tools/review_wing_readonly_preflight_artifacts.py` gives the operator a
  direct artifact-review command and clear exit codes.

`tools/no_wing_release_check.py` already surfaced the preflight evidence status,
but it did not prove that the capture/review CLIs still existed or show the
operator next commands in its JSON/Markdown summary.

## Options considered

1. Leave the review command only in `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
2. Run the review CLI automatically from `no_wing_release_check.py`.
3. Keep `no_wing_release_check.py` offline and non-executing, but require the
   capture runner and review tool to exist and publish the capture command
   template, review command, and review exit-code contract in release summaries.

## Decision

Use option 3. The no-WING release gate now fails closed if
`tools/wing_readonly_preflight_runner.py` or
`tools/review_wing_readonly_preflight_artifacts.py` is missing. Its
`wing_readonly_preflight_evidence` JSON and Markdown summary sections include:

- `capture_tool_path`
- `capture_tool_available`
- `capture_command_template`
- `capture_command_requires_operator_channel_confirmation`
- `capture_command_requires_physical_readback`
- `review_tool_path`
- `review_tool_available`
- `review_command`
- `review_exit_codes`

## Why this won

The selected MVP blocker is still WING read-only connection proof. The release
summary should point the operator to the exact artifact review step without
contacting hardware or treating offline evidence as live readiness.

Automatically running the review CLI would be convenient, but the no-WING gate
is a broad release check and should not silently convert artifact state into a
readiness closure. Publishing the command keeps the manual review boundary
visible.

## Rejected alternatives

Docs-only guidance can drift from the release gate. Auto-running the review CLI
inside the release check risks implying that a clean candidate closes readiness,
even though real WING artifacts still require manual review.

## Implementation plan

1. Add review-tool constants and a command builder to `tools/no_wing_release_check.py`.
2. Fail `validate_wing_readonly_preflight_evidence_snapshot()` if the review
   tool path is missing.
3. Propagate capture and review command metadata into
   `wing_readonly_preflight_evidence_summary()`.
4. Render the same metadata in `summary.md`.
5. Add tests for normal metadata, missing-tool failure, and JSON/Markdown
   summary rendering.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- Focused `tests/test_no_wing_release_check.py` cases for preflight evidence
  metadata and missing review tool.
- Adjacent preflight/MVP tests.
- Quick no-WING release-check CLI run with `/tmp` artifact output.
- Full pytest suite before handoff.

## Risks and rollback

Risk: a local checkout without the review CLI now fails the release gate. This
is intentional; the selected WING MVP blocker depends on that review path.

Rollback is a small revert of the constants, missing-tool violation, summary
fields, and tests. No mixer state is touched.
