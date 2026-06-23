# WING MVP no-WING Operator Markdown Summary SHA Match

## Summary

The no-WING `next_wing_proof_plan` summary now records the actual SHA256 for the
operator-readable `summary.md` text and a match verdict against the declared
`operator_markdown_summary_sha256`.

## Changed

- `tools/no_wing_release_check.py` adds an artifact-integrity helper for the
  operator markdown summary binding.
- Generated summaries expose `operator_markdown_summary_actual_sha256` and
  `operator_markdown_summary_sha256_matches_actual`.
- A stale markdown digest fails closed in `tests/test_no_wing_release_check.py`.
- Project Director rails require the match flag before supervised proof
  planning.

## Safety Boundary

This is an offline no-WING artifact integrity guard. It does not contact WING,
query a console, mutate mixer state, grant live authority, or authorize
fader/gain movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
