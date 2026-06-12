# Codex Proposal: WING Read-Only no-WING Current Backend-Port Blocker

## Short thesis

Record the current no-WING quick gate result as a focused evidence report
instead of changing code that already exposes the required busy-port diagnostics.

## What I understood

The current MVP rail starts with WING read-only connection proof. The required
first command is the no-WING quick gate before starting an intended backend/WING
session. In this worktree, the gate still fails closed because
`127.0.0.1:8765` has an existing Python listener that requires operator
confirmation.

## Proposed solution

Add a dated `Docs/reports/` note that records:

- the exact command run;
- the failed check;
- the listener identity fields reported by the gate;
- the safety boundary;
- the operator next step.

## Likely files to touch

- `.ai/briefs/wing-readonly-no-wing-current-backend-port-blocker-20260612.md`
- `.ai/proposals/codex-wing-readonly-no-wing-current-backend-port-blocker-20260612.md`
- `Docs/adr/wing-readonly-no-wing-current-backend-port-blocker-20260612.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_NO_WING_CURRENT_BACKEND_PORT_BLOCKER_20260612.md`

## Alternatives considered

- Stop the listener automatically. Rejected: the no-WING gate explicitly says
  the operator must confirm whether it is the intended backend first.
- Change the backend-port gate. Rejected: the current gate already exposes the
  checklist, process list, fail-closed reason, and operator next step.
- Commit the generated timestamped artifact directory. Rejected: the operator
  report is sufficient evidence for this source-control slice, and the
  timestamped artifact is local run output.

## Risks

- A report-only change could be mistaken for readiness. Mitigation: the report
  explicitly says it closes no blocker and grants no live authority.

## Test plan

- Run the no-WING quick gate with `--continue-on-failure`.
- Run repository lint/check commands that do not require hardware.
- Review the git diff before commit.

## Where the other agent may disagree

Kimi may argue that no report should be committed without a code change. The
counterpoint is that this report captures the current first-blocker state after
running the required command and prevents stale handoff assumptions.
