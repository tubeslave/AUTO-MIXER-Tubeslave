# Brief: WING Read-Only Preflight Decision Artifact Hash Binding - 2026-06-07

## Context

The first supervised WING MVP blocker remains the real read-only connection
proof. The current flow captures `summary.json` and `events.jsonl`, reviews the
artifact offline, then records an operator decision JSON before supervised proof
planning can proceed.

## Problem

The review decision ledger matched decisions to a candidate by source id,
`run_id`, `summary_path`, and `events_path`. Those fields identify the file
locations, but they do not prove the operator decision still matches the exact
file contents after an artifact is replaced at the same path.

## Scope

- Record SHA256 digests for `summary.json` and `events.jsonl` in evidence rows.
- Include those digests in pending review decision templates.
- Reject accepted or rejected decision records whose digests are missing or
  stale.
- Update the runbook, no-WING gate expectations, tests, ADR, and report.

## Out Of Scope

- Running WING hardware.
- Changing readiness rows from offline evidence.
- Granting live authority or widening supervised writes.
