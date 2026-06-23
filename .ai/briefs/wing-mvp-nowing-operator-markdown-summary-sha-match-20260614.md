# Brief: WING MVP no-WING operator markdown summary SHA match

## Task

Make the no-WING proof-plan summary report whether its declared
operator-readable `summary.md` SHA256 matches the actual markdown text.

## Context

The previous slice bound `summary.json` to `summary.md` with a JSON-only digest.
The WING handoff should also expose a machine-readable match verdict so stale or
incorrect markdown digests fail closed before supervised proof planning.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority.
- Keep the digest JSON-only to avoid recursive markdown hashing.
- Keep Project Director focused on the first WING blocker.

## Acceptance

- `next_wing_proof_plan` records `operator_markdown_summary_actual_sha256`.
- `next_wing_proof_plan` records
  `operator_markdown_summary_sha256_matches_actual: true` for generated
  summaries.
- A stale declared markdown SHA fails the summary closed in tests.
- Project Director Current First Command baseline requires the match verdict.
