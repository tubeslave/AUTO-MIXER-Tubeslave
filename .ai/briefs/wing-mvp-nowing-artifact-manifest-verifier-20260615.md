# Brief: WING MVP no-WING Artifact Manifest Verifier

## Context

The no-WING release check writes `summary.json`, operator-readable
`summary.md`, and detached `artifact-manifest.json`. The sidecar now carries
path, SHA256, cross-surface, self-consistency, and no-authority verdicts, but an
operator still needs a cheap way to verify the bundle before opening a real
backend/WING session.

## Goal

Add a read-only CLI verifier for a generated no-WING artifact directory. It
must recompute the bundle hashes and existing manifest contract verdicts, fail
closed on drift, and never contact WING or grant live authority.

## Non-Goals

- Do not change WING runtime behavior.
- Do not close the WING read-only connection proof blocker.
- Do not approve supervised writes.
- Do not add production dependencies.

## Test Plan

- Focused verifier tests for a valid bundle and stale `summary.md`.
- `py_compile` for the new tool and no-WING tests.
- no-WING quick run followed by verifier CLI.
- Focused WING/no-WING/Project Director tests and full pytest.
