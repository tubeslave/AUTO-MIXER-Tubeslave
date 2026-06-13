# Proposal: WING MVP Busy-Port Prestart Command Summary

## Short thesis

Surface the shared backend-port prestart check command directly in failed
busy-port no-WING summaries, so the first fail-closed artifact tells the
operator exactly which no-WING gate must pass before starting a real WING
session.

## What I understood

The current first blocker remains `WING read-only connection proof`. Because no
real WING hardware is available here, progress should strengthen the no-WING
prestart evidence without treating it as readiness. The handoff and proof-plan
contracts already pin `backend_port_prestart_check_command`, but a failed
`backend_port_free` summary does not currently show that command itself.

## Proposed solution

- Populate `backend_port_prestart_check_command` in the failed
  `backend_port_free` detail whenever an existing listener is detected.
- Render that command in `summary.md`.
- Add tests for JSON and Markdown output.
- Update WING MVP docs and autoloop rails to require the command in both the
  failed busy-port no-WING summary and no-WING handoff summary.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Only require the command in handoff packets. That keeps later checks
consistent, but it leaves the first failed prestart artifact less actionable
for operator review.

## Risks

The command is diagnostic and rerun-oriented only. It must not be read as
permission to start the backend, stop listeners, contact WING hardware, or
close readiness.

## Test plan

- Compile the touched no-WING module.
- Run focused no-WING, WING MVP rail, Project Director, and direction guard tests.
- Run the no-WING quick gate and inspect saved `summary.md` plus `summary.json`.
- Run memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may argue that repeating the command in a failed summary is redundant
because downstream handoff checks already compare it. The counterpoint is that
the first fail-closed artifact should be self-contained enough for a safe
operator handoff.
