# Proposal: WING MVP Busy-Port Prestart Required Flags Summary

## Short thesis

Surface the backend-port prestart required and must-pass flags directly in
failed busy-port no-WING summaries, so the first fail-closed artifact states
that the no-WING backend-port gate is mandatory before a real WING session.

## What I understood

The current first blocker remains `WING read-only connection proof`. Because no
real WING hardware is available here, progress should strengthen the no-WING
prestart evidence without treating it as readiness. The handoff summary already
requires `backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true`, but a
failed `backend_port_free` summary does not currently show those flags itself.

## Proposed solution

- Populate `backend_port_prestart_check_required: true` and
  `backend_port_prestart_check_must_pass_before_starting_backend: true` in the
  failed `backend_port_free` detail whenever an existing listener is detected.
- Render those flags in `summary.md`.
- Add tests for JSON and Markdown output.
- Update WING MVP docs and autoloop rails to require the flags in both the
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

Only require the flags in handoff packets. That keeps later checks consistent,
but it leaves the first failed prestart artifact less explicit about the
non-optional gate boundary.

## Risks

The flags are fail-closed metadata only. They must not be read as permission to
start the backend, stop listeners, contact WING hardware, or close readiness.

## Test plan

- Compile the touched no-WING module.
- Run focused no-WING, WING MVP rail, Project Director, and direction guard tests.
- Run the no-WING quick gate and inspect saved `summary.md` plus `summary.json`.
- Run memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may argue that handoff summaries already carry the required flags. The
counterpoint is that the first fail-closed artifact is what the operator sees
when the prestart gate blocks, so it should carry the same mandatory-boundary
contract.
