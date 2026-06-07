# Codex Proposal: WING Read-Only Preflight Offline Plan Preview

## Short thesis

Add a non-network `--plan-only` mode to the WING read-only preflight runner and
wire it into the operator handoff/no-WING checks before the hardware capture
step.

## What I understood

The next live-readiness barrier is still real WING read-only evidence, not more
offline claims. The safest useful improvement is to make the exact planned
request sequence visible and testable before any WebSocket session opens.

## Proposed solution

- Extend `tools/wing_readonly_preflight_runner.py` with `--plan-only` and
  `--plan-output`.
- Emit an offline JSON plan containing channels, value types, operator
  confirmation, intended read-only requests, forbidden request types, expected
  artifacts, and false authority flags.
- Include the selected MVP blocker, manual review requirement, fail-closed
  conditions, and post-run review commands in the plan JSON.
- Add `preview_readonly_request_plan` to the handoff packet.
- Run the plan preview as a no-WING smoke command.
- Fail no-WING schema checks if the preview command loses `--plan-only`, lacks
  output, or drifts from the capture command.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-offline-plan-preview-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OFFLINE_PLAN_PREVIEW_20260607.md`
- `Docs/INDEX.md`

## Alternatives considered

1. Keep only the real capture command in the runbook.
2. Put preview text only in docs.
3. Add a separate preview script.

## Risks

- Operators might mistake an offline plan for hardware proof.
- The preview command could drift from the real capture command.
- no-WING checks could silently stop exercising the preview path.

## Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- Focused pytest for preflight runner, handoff packet, and no-WING checks.
- no-WING quick smoke.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may prefer a separate preview tool. Keeping the preview inside the existing
runner reduces drift because the plan uses the same argument parser and read
request builder as the real read-only capture path.
