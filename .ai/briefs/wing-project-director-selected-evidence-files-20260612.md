# Brief: WING Project Director Selected Evidence Files

Date: 2026-06-12

## Task

Keep autonomous Codex cycles focused on the first supervised WING MVP blocker by
surfacing the selected blocker evidence files directly in the Project Director
directive.

## Context

The first open blocker, `WING read-only connection proof`, already lists the
tools, contracts, and reports that define the read-only preflight gate. Recent
cycles had to add missing required files manually. The Director should preserve
that blocker-row evidence as machine-readable directive data so future cycles
inspect the proof files already named by the blocker table.

## Safety Boundary

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No listener stop or reconfiguration.
- No live authority or readiness closure from offline evidence.

## Expected Outcome

Project Director JSON and Markdown expose `selected_evidence_files` and a count
for the first open blocker, while required runtime files may be derived from
tool/backend/runbook evidence without promoting historical reports into required
source files.
