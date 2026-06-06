# Brief: Read Surface Fixture Acceptance Template

## Context

The read-surface physical readback chain now has capture planning, an operator
capture packet, per-surface artifact intake, an aggregate review queue, and a
fixture acceptance ledger. The remaining handoff gap is a concrete operator
template for turning a valid review candidate into an acceptance/rejection
record.

## Goal

Add a transport-free offline template builder that creates pending acceptance
records only from valid review candidates and keeps all fixture-closure and
live-authority flags false.

## Constraints

- Do not query WING, dLive, OSC, MIDI, TCP, websockets, or mixer clients.
- Do not close fixture gaps.
- Do not update readiness rows.
- Do not grant live authority.
- A generated template must require operator edits before the ledger accepts it.

## Validation

- Unit tests for missing candidates, valid candidate templates, ledger
  compatibility after edits, CLI output, and transport-free source.
- no-WING quick-mode smoke command for the template builder.
