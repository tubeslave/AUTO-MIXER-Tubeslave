# Brief: WING MVP Supervised Prestart Artifact-Root Marker Gate

## Task

Make the final supervised prestart gate explicitly require a valid
accepted-candidate artifact-root trace before any WING supervised fader/gain
proof planning.

## Context

The ledger, no-WING gate, decision-review tool, and supervised proof plan already
enforce accepted-candidate artifact-root validity. The remaining risk was a
human-facing contract mismatch: the final supervised prestart gate text and
baseline markers still named the accepted-candidate prestart trace but did not
name the artifact-root trace.

## Safety Boundary

This is an offline contract/reporting change. It does not contact WING, mutate
console state, grant live authority, or close readiness.

## Expected Evidence

- Contract constants include artifact-root trace as a required marker.
- Handoff/no-WING/Project Director surfaces expose the same marker.
- Runbook and MVP docs describe the same blocker.
- Focused WING/no-WING tests pass.
