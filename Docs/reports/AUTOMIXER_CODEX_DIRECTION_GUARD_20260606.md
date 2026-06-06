# Automixer Codex Direction Guard - 2026-06-06

## Summary

This slice adds repo-local automation that checks whether Codex work is moving
toward the current supervised WING Rack MVP.

The guard:

- reads git status, recent commit subjects, and recent repo-local reports;
- scores target evidence against WING proof blockers;
- flags drift toward memory-trust, queue-import, result-consumer, report-only,
  broad read-model, or proposal-only work;
- recommends the next allowed WING proof actions;
- runs read-only and never touches WING/OSC/WebSocket transport.

## Activation Time

The automation is active inside each Codex autoloop cycle:

1. before Codex chooses the cycle task;
2. before Codex writes the final cycle report.

When the existing LaunchAgent is installed, the wall-clock cadence remains the
existing 1200-second interval. No new system LaunchAgent is installed by this
change.

## Safety Boundary

- No live WING/OSC action is run.
- No real console is contacted.
- The guard does not modify code.
- The guard does not grant live authority or readiness closure.

## Files Changed

- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_codex_direction_guard.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `.ai/briefs/codex-direction-guard-20260606.md`
- `.ai/proposals/codex-codex-direction-guard-20260606.md`
- `Docs/adr/codex-direction-guard-20260606.md`

## Validation

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py` - PASS
- `python3 tools/codex_autoloop/codex_direction_guard.py --format json --output .paperclip/reports/codex_direction_guard_latest.json` - PASS
- `python3 tools/codex_autoloop/codex_direction_guard.py --format markdown --output .paperclip/reports/codex_direction_guard_latest.md` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py tests/test_codex_cycle_prompt.py tests/test_codex_autoloop_contract.py` - `20 passed`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

Current guard result for the live dirty worktree: `mixed_needs_review`, score
`30`. That means the current Codex work contains real WING-proof progress but
still carries drift signals from memory/queue/result-consumer style work.
