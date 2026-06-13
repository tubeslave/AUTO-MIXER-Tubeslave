# Proposal: WING MVP Busy-Port Prestart Detail Fields Summary

## Short thesis

Surface the shared backend-port prestart failure-detail field list directly in
failed busy-port no-WING summaries, so the first artifact an operator sees is
already aligned with the handoff/proof-plan contract.

## What I understood

The current first blocker remains `WING read-only connection proof`. Because no
real WING hardware is available here, progress should strengthen the no-WING
prestart evidence without treating it as readiness. The busy-port gate already
fails closed and emits individual detail fields, but the shared field list is
not visible in the failed `backend_port_free` result.

## Proposed solution

- Populate `backend_port_prestart_check_failure_detail_fields` in the
  `backend_port_free` detail whenever an existing listener is detected.
- Render that list in `summary.md`.
- Add tests for JSON and Markdown output.
- Update WING MVP docs and autoloop rails to require the field list in both the
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

Only require the field list in handoff packets. That keeps later checks
consistent, but it leaves the first failed prestart artifact less explicit than
the rest of the WING read-only path.

## Risks

The field name includes "failure" and may still appear if an operator confirms
the existing listener. That is acceptable because it describes the prestart
failure-detail contract, not live authority.

## Test plan

- Compile the touched no-WING module.
- Run focused no-WING, WING MVP rail, Project Director, and direction guard tests.
- Run the no-WING quick gate and inspect saved `summary.md` plus `summary.json`.
- Run memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may argue that this is duplicative because the detail fields themselves
are already present. The counterpoint is that the shared exact list is what
handoff/proof-plan consumers compare, so the first failed prestart artifact
should expose the same contract.
