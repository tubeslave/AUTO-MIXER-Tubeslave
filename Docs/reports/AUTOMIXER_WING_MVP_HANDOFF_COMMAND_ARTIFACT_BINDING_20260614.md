# WING MVP Handoff Command Artifact Binding

## Summary

The WING read-only handoff summary will expose command-level bindings proving
that capture, artifact review, and decision review commands point to the same
expected plan, artifact root, and decision root as the packet before any real
WING session.

The no-WING handoff summary must show:

- `preview_command_artifact_root_matches_packet: true`
- `preview_command_plan_output_matches_packet: true`
- `capture_command_artifact_root_matches_packet: true`
- `capture_command_expected_plan_matches_packet: true`
- `review_command_artifact_root_matches_packet: true`
- `decision_review_command_artifact_root_matches_packet: true`
- `decision_review_command_decision_root_matches_packet: true`

This is no-WING handoff hardening. It does not contact WING, start a backend,
write mixer state, grant live authority, or close the WING read-only connection
proof blocker.

## Validation

Completed before commit:

- Touched Python compile check: passed.
- Focused no-WING/Project Director/WING rail tests: `263 passed`.
- no-WING quick release check with backend port `65535`: passed.
- Project Director stayed on `WING read-only connection proof` with
  `guard_status=on_target`, `guard_score=90`, and `guard_stop_the_line=False`.
- Direction guard stayed `on_target`, `score=90`, and
  `stop_the_line=False`.
- Full pytest: `3288 passed, 1 skipped, 1 warning`.
