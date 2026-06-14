# Brief: WING MVP Handoff Supervised Summary Field Contract

## Task

Make the WING read-only handoff packet declare the expected supervised
proof-plan summary field contract and its match verdict.

## Context

The no-WING handoff summary already renders
`expected_required_supervised_proof_plan_summary_fields` and
`required_supervised_proof_plan_summary_fields_match_contract`. The handoff
packet itself only declared `required_supervised_proof_plan_summary_fields`,
so stale packet declarations could be detected only after no-WING recomputed
the summary.

## Safety Boundary

This is offline handoff metadata and validation work. It must not start a
backend, contact WING, query a console, write to a console, grant live
authority, update readiness, or close `WING read-only connection proof`.

## Deliverable

Add the declared expected field list and match flag to the packet, render them
in packet Markdown, make no-WING validation reject stale declarations, and add
focused tests.
