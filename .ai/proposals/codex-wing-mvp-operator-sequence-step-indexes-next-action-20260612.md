# Codex Proposal: WING MVP Operator Sequence Step Indexes Next Action

## Short Thesis

Name the digest-review and supervised-gate operator sequence indexes in the
selected first-blocker action and mirror them across runbook, MVP, autoloop, and
test surfaces.

## What I Understood

The handoff validator already has structured sequence information, but the
operator-facing selected action does not explicitly require the two non-gate
operator sequence index fields. That leaves reviewers checking step content
without a named positional invariant.

## Proposed Solution

- Add `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`.
- Add `operator_sequence_supervised_prestart_gate_step_index: 9`.
- Place them in the first WING blocker next action and current command.
- Mirror the same requirement in the read-only preflight runbook, MVP rail,
  autoloop prompt, and autoloop README.
- Add tests so Project Director, generated progress, runbook, MVP doc, prompt,
  README, and docs index retain the fields.

## Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/INDEX.md`

## Alternatives Considered

- Change validator logic now: unnecessary for this slice because the issue is
  missing selected-action/runbook visibility.
- Wait for hardware: hardware is still required for proof closure, but this
  repo-local rail can reduce operator review ambiguity before hardware.

## Risks

- Documentation drift if only one surface is updated.
- Overstating readiness from offline handoff data. The text must keep stating
  that the packet grants no live authority and does not close the blocker.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that more docs-only rails add process weight. The counterpoint is
that this rail is narrow, directly tied to the first WING blocker, and makes the
handoff sequence easier to reject when it is stale or reordered.
