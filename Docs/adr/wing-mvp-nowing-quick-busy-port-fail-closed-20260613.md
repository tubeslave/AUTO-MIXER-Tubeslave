# ADR: WING MVP No-WING Quick Busy-Port Fail-Closed

## Context

The first supervised WING MVP blocker is still `WING read-only connection
proof`. The required first operator command is the no-WING quick gate before
starting the intended backend/WING session. On 2026-06-13 that gate failed
closed because a Python listener was already bound to `127.0.0.1:8765`.

## Options considered

1. Leave the generated no-WING artifact untracked.
2. Track the artifact and report it as current first-blocker evidence.
3. Automatically stop or reconfigure the listener.

## Decision

Use option 2.

## Why this won

The artifact is the authoritative current evidence for the local prestart gate:
the system did not contact WING, did not grant live authority, and required
manual operator confirmation before continuing. Tracking it keeps Project
Director aligned with the real next action.

## Rejected alternatives

- Option 1 loses the current fail-closed evidence after the turn.
- Option 3 violates the no-WING gate: the gate may inform the operator, but it
  must not stop or reconfigure a listener on its own.

## Implementation plan

- Preserve the no-WING `summary.json` and `summary.md` artifact.
- Add an operator-visible report.
- Add that report to the first WING blocker evidence list.
- Assert blocker parsing and Project Director selection include the report.
- Regenerate the latest MVP progress report.

## Test plan

- Focused WING rails, Project Director, and Direction Guard tests.
- Project Director JSON check.
- Direction Guard JSON check.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the report, artifact, and evidence references. It does not
change WING runtime behavior or operator authority.
