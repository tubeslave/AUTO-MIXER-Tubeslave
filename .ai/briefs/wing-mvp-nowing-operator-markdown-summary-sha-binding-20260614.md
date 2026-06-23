# Brief: WING MVP no-WING operator markdown summary SHA binding

## Task

Bind the generated no-WING `summary.json` proof-plan summary to the exact
operator-readable `summary.md` artifact bytes.

## Context

The WING MVP still waits for real read-only WING proof and an accepted operator
review decision. Offline no-WING summaries are handoff artifacts only; they must
not grant live authority. The previous slice made the markdown proof-plan surface
self-checking, but `summary.json` did not yet identify the exact markdown file an
operator reviewed.

## Constraints

- Do not contact WING or backend services.
- Do not add live-write authority.
- Avoid recursive markdown hashing by keeping the markdown SHA in JSON only.
- Keep Project Director focused on the first WING blocker.

## Acceptance

- Generated no-WING `summary.json` includes `operator_markdown_summary_artifact:
  summary.md`.
- Generated no-WING `summary.json` includes a SHA256 for the actual
  `summary.md` bytes and marks it available.
- Project Director Current First Command baseline requires the artifact and SHA
  fields before supervised proof planning.
- Focused no-WING and WING rail tests pass.
