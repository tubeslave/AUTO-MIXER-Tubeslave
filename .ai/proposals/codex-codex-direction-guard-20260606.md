# Proposal: Codex Direction Guard

## Short thesis

Add a read-only direction guard to the Codex autoloop so each cycle is checked
against the supervised WING Rack MVP blockers before it chooses work and before
it writes the final report.

## What I understood

The project needs Codex to keep moving toward real WING proof, not just produce
more safe but indirect memory/queue/report abstractions.

## Proposed solution

- Add `tools/codex_autoloop/codex_direction_guard.py`.
- Classify recent git status, commit subjects, and reports as target evidence,
  drift evidence, or mixed evidence.
- Add tests for WING-proof work, memory/queue drift, and mixed cycles.
- Update `codex_cycle_prompt.md` so each cycle runs the guard before planning
  and before final reporting.

## Likely files to touch

- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_codex_direction_guard.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`

## Alternatives considered

- Install a new LaunchAgent. Rejected: the repo already has a 20-minute
  LaunchAgent contract, and installing system automation is riskier than
  adding repo-local checks.
- Make the guard edit code automatically. Rejected: the guard must observe and
  steer, not become another autonomous writer.

## Risks

- Keyword classification can be imperfect.
- A mixed cycle may contain both useful WING work and old drift artifacts.

## Test plan

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py`
- Existing prompt/autoloop contract tests.

## Where the other agent may disagree

Another agent may prefer enforcing the guard in the shell wrapper. The safer
first step is prompt-level enforcement plus an explicit command, because it
does not change launchd behavior or background execution.
