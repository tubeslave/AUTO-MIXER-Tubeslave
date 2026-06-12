# Codex Proposal: wing-project-director-handoff-required-file-20260612

## Short thesis

Project Director should list the handoff packet builder as a required file
because current proof planning depends on the structured handoff gates it emits.

## What I understood

The first blocker is still `WING read-only connection proof`. The useful offline
work is to keep automation pointed at the exact read-only proof path, not to
claim readiness or widen writes.

## Proposed solution

- Add `tools/build_wing_readonly_preflight_handoff_packet.py` to
  `project_director_agent.py` `required_files`.
- Extend the Project Director unit test to assert that requirement.
- Record the planning-only safety boundary in ADR/report artifacts.

## Likely files to touch

- `tools/codex_autoloop/project_director_agent.py`
- `tests/test_project_director_agent.py`
- `Docs/adr/wing-project-director-handoff-required-file-20260612.md`
- `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md`

## Alternatives considered

- Leave the builder implicit through `tools/no_wing_release_check.py`. Rejected:
  the current selected action names handoff summary gates, and the builder is
  the direct source for that contract.
- Add broad docs instead. Rejected: Project Director required files are the
  automation-facing surface that chooses what future cycles must inspect.

## Risks

The list could become too large. This addition is narrow and directly tied to
the selected blocker handoff gate.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may say the no-WING release checker already validates the handoff packet.
I would still expose the builder because Project Director required files are
for human/agent inspection before planning, not only automated validation.
