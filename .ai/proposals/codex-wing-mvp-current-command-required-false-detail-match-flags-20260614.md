# Codex Proposal: WING MVP Current Command Required-False Detail Match Flags

## Short thesis

Promote the backend-port required-false detail match verdicts into the Current
First Command baseline so Goal-mode cycles cannot treat a no-WING handoff as
complete unless packet and plan agree on the no-authority fields.

## What I understood

The WING read-only connection proof is still blocked on real hardware evidence.
Offline handoff work is only useful if it preserves the exact safety boundary:
no stop permission, no live authority, and no backend-port gate authority. The
validator already emits match verdicts; the selected action must require them.

## Proposed solution

- Add `safe_to_stop_listener_by_gate_matches_plan: true`,
  `live_authority_granted_matches_plan: true`, and the three
  `backend_port_prestart_gate_*_matches_plan: true` verdicts to
  `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add the same markers to `Current First Command`, the first blocker next
  action, MVP docs, runbook, and autoloop prompt/readme.
- Extend existing marker-loop tests rather than adding a new test surface.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md` after the marker count
  changes.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/project_director_agent.py` callers through the shared
  marker set
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## Alternatives considered

- Keep the match verdicts only in the no-WING validator output. Rejected because
  Project Director follows the selected action baseline, not raw validator
  internals.
- Add the markers only to docs. Rejected because the baseline marker list should
  fail if Current First Command omits these fields.

## Risks

The marker count changes and stale Current First Command copies will fail
coverage tests. That is intended; stale rails should be refreshed before the
next WING-prep cycle.

## Test plan

- Focused Project Director and WING rail pytest.
- Project Director JSON inspection.
- no-WING quick check.
- memory lint and `git diff --check`.
- Standard full pytest when time allows.

## Where the other agent may disagree

Kimi may prefer less repetition in operator docs. I would keep the repetition
because these are safety-critical no-authority verdicts and omission changes the
meaning of the handoff.
