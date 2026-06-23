# Master / Output PA-feed-safe Proposal Layer

Date: 2026-06-15
Status: implemented

## Task

Move Master / Output from `offline_method_needed` to a concrete proposal-only
output recommendation layer while preserving the current WING live-write
boundary.

## Constraints

- Do not contact WING, OSC, MIDI, or mixer clients.
- Do not queue or apply master, matrix, limiter, EQ, compressor, or output-route
  writes.
- Do not treat reference/measurement feeds as writable PA outputs.
- Require true-peak headroom and PA/output-feed safety evidence before a
  proposal-only recommendation can be emitted.
- Keep all master/output work non-queueable until physical readback, rollback,
  and hardware proof are complete.

## Acceptance

- `master_bus_optimization_proposal` remains transport-free and proposal-only.
- Missing PA-feed readback, missing PA-feed safety, stale PA-feed freshness, and
  reference-feed roles block proposal-only output.
- Proposal contracts, registry, capability matrix, backlog, ADR, report, and
  tests reflect the boundary.
