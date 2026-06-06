# ADR: Codex Direction Guard

## Context

Recent Codex cycles improved safety, but some cycles drifted into memory,
queue-import, result-consumer, and report-only layers. The current project
priority is supervised WING Rack MVP proof: read-only preflight, physical
readback, supervised fader/gain proof, rollback, emergency stop, timeout
fail-closed, and readiness updates based on real proof.

## Options considered

1. Add a repo-local direction guard used by each Codex cycle.
2. Install a new macOS background service.
3. Rely on prompt wording only.

## Decision

Use option 1.

## Why this won

A repo-local guard is auditable, testable, and safe. It can steer Codex without
installing another daemon or granting live mixer authority.

## Rejected alternatives

- A new background service would add operational risk and could conflict with
  the existing 20-minute autoloop LaunchAgent contract.
- Prompt wording alone helped, but it does not produce an explicit direction
  audit artifact.

## Implementation plan

1. Add `tools/codex_autoloop/codex_direction_guard.py`.
2. Add focused tests.
3. Update the autoloop prompt to run the guard before task selection and before
   the final report.
4. Document activation timing in the autoloop README.

## Test plan

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: keyword classification may over-warn on mixed but useful work. Mitigation:
the guard reports evidence and recommendations instead of editing code or
blocking by default. Rollback is removing the guard script, tests, and prompt
hook.
