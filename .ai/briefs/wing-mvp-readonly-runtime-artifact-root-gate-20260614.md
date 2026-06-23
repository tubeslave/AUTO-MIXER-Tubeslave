# Brief: WING MVP read-only Runtime Artifact Root Gate

Date: 2026-06-14
Task ID: wing-mvp-readonly-runtime-artifact-root-gate-20260614

## Context

The WING read-only runner already validates a real run against an offline
`--expected-plan`, and the offline plan declares the artifact root used for
`summary.json` and `events.jsonl`. The captured real-run summary did not
explicitly record the runtime `--artifact-root`, leaving reviewers to infer it
from paths instead of checking it as part of the plan-vs-runtime contract.

## Goal

Make the real read-only preflight summary declare the runtime artifact root and
make evidence intake fail closed when the summary omits it or when the
referenced offline plan declares a different artifact root.

## Safety Boundary

- No WING hardware contact.
- No backend/WING WebSocket opening in validation.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused read-only runner and evidence tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Memory lint and whitespace diff.
