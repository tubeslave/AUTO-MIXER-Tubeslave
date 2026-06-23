# WING MVP no-WING Operator Markdown Summary SHA Binding

## Summary

The no-WING `summary.json` handoff now binds `next_wing_proof_plan` to the exact
operator-readable `summary.md` artifact through a JSON-only artifact name and
SHA256 digest.

## Changed

- `tools/no_wing_release_check.py` records
  `operator_markdown_summary_artifact: summary.md`,
  `operator_markdown_summary_sha256`, and
  `operator_markdown_summary_sha256_available: true` after final markdown
  rendering.
- `tests/test_no_wing_release_check.py` hashes the actual written `summary.md`
  and verifies the JSON digest matches it.
- Project Director Current First Command, prompt, README, and rail tests now
  require the artifact and digest markers before supervised proof planning.

## Safety Boundary

This is an offline no-WING artifact integrity guard. It does not contact WING,
query a console, mutate mixer state, grant live authority, or authorize
fader/gain movement.

## Remaining Blocker

The WING MVP still requires a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
