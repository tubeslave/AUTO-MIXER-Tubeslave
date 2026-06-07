# Brief: Read-Surface Acceptance Candidate Trace

## Task

Preserve the review-queue candidate trace inside read-surface physical readback
acceptance-template records and make the acceptance ledger reject records whose
trace is missing or no longer matches the queue candidate.

## Context

The review queue now preserves artifact source ids for valid physical-readback
candidates. The downstream acceptance template includes `artifact_source_id`,
but a manually edited acceptance record still needs an auditable snapshot of the
queue candidate it was generated from: queue status, candidate status, artifact
source ids, candidate counts, and the fail-closed authority flags.

## Safety Boundary

- Do not query or mutate WING, dLive, OSC, MIDI, TCP, backend WebSocket, or any
  console transport.
- Do not accept physical readback into fixtures automatically.
- Do not update fixture baselines or close physical fixture gaps.
- Keep live widening, live authority, and full autonomous readiness false.
- Treat missing or mismatched candidate trace as replacement required.

## Acceptance Criteria

- Acceptance-template records include a `candidate_trace` copied from the
  review-queue candidate.
- The ledger rejects operator records when `candidate_trace` is absent or
  diverges from the queue candidate.
- no-WING release checks expose candidate-trace counts and fail closed on trace
  gaps.
- Reports and ADRs keep this framed as a read-only operator gate, not live
  readiness.
