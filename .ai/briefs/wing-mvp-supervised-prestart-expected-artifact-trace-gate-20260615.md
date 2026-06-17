# WING MVP Supervised Prestart Expected Artifact Trace Gate

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The handoff operator step already requires an `expected-artifact-bound accepted
source id`, and the proof plan now keeps
`accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`
in force when accepted-candidate expected artifact trace evidence is incomplete.

## Task

Make the structured `supervised_prestart_gate` explicitly require
`requires_accepted_candidate_expected_artifact_trace: true` before any
supervised fader/gain proof planning can be considered aligned.

## Constraints

- Offline contract and validation surfaces only.
- Do not contact WING, start a backend, or query a console.
- Do not grant live authority or readiness closure.
- Preserve the existing prestart and artifact-root trace requirements.

## Test Plan

- Focused contract/no-WING/director tests.
- no-WING quick check.
- Full pytest if the focused suite is green.
- `tools/memory/memory_lint.py`.
- `/opt/homebrew/bin/git diff --check`.
