# Codex Proposal: WING Read-Only Preflight Event-Log Crosscheck

## Short thesis

Accept read-only preflight filesystem artifacts only when `summary.json` and
`events.jsonl` agree on the actual request/readback flow.

## What I understood

The current WING Rack MVP is blocked on real read-only connection and physical
readback proof before supervised fader/gain tests. The preflight runner writes
both summary and event log artifacts, but evidence intake has been validating
the summary alone.

## Proposed solution

- When validating a discovered `summary.json`, resolve and load its event log.
- Require contiguous event `seq` values, matching the runner's append-only
  recorder shape.
- Compare event-log send request types with summary `sent_request_types`.
- Reject event logs with write/proposal/operator-mode/rollback/emergency
  request types.
- Compare event-log readback payloads with summary `readbacks`.
- Keep inline payload validation available for unit-level tests and synthetic
  snapshots.

## Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `.ai/briefs/`
- `Docs/adr/`
- `Docs/reports/`
- `.ai/memory/project.md`

## Alternatives considered

1. Trust `summary.json` only.
2. Require a cryptographic hash between summary and events.
3. Crosscheck the event log structurally without adding crypto.

Option 3 is the smallest useful safety step. Hashing can come later if artifact
handoff becomes a real operator workflow problem.

## Risks

Copied summaries without their event logs now fail closed. That is acceptable
because a readiness-review candidate should be based on the complete runner
artifact.

## Test plan

- Valid filesystem artifact with matching event log remains a candidate.
- Missing event log fails closed.
- Event log with a forbidden send request fails closed.
- Event log with missing or non-contiguous sequence numbers fails closed.
- Existing readiness, proof-plan, release-check, and runner tests remain green.

## Where the other agent may disagree

Kimi may argue for signed artifact bundles instead of structural checking. That
is stronger but larger than needed for the current preflight gate.
