# ADR: WING Read-Only Preflight Offline Plan Preview

## Context

The supervised WING MVP is still blocked on real read-only connection and
physical readback proof. The repository has a preflight runner and handoff
packet, but operators need a way to inspect the exact request sequence before a
backend/WING WebSocket session opens.

## Options considered

1. Keep only the real hardware capture command.
2. Add documentation-only request examples.
3. Add `--plan-only` mode to the existing preflight runner and verify it in
   no-WING checks.

## Decision

Use option 3. `tools/wing_readonly_preflight_runner.py` now supports
`--plan-only` and `--plan-output`. Plan-only mode validates the same arguments
as the real capture path and emits the intended read-only request sequence
without opening WebSocket, querying WING, or writing physical proof artifacts.
The no-WING release check also reads the generated plan JSON back and validates
its schema, safety flags, expected request sequence, and output path.

## Why this won

The plan preview reduces operator error at the exact WING proof boundary while
keeping the real proof requirement intact. Using the same runner avoids a
second request builder that could drift from the hardware path.

## Rejected alternatives

- Capture-only runbooks leave channel-confirmation mistakes to be found during
  the live backend/WING session.
- Documentation-only examples can drift from the command parser and request
  builder.
- A separate preview tool would duplicate read-request generation.

## Implementation plan

1. Add `--plan-only` and `--plan-output` to the preflight runner.
2. Emit an offline plan with read-only requests, false live-authority flags,
   forbidden request types, and expected hardware artifacts.
3. Add `preview_readonly_request_plan` to the handoff packet.
4. Run the preview as a no-WING smoke command.
5. Fail schema validation if preview channels, confirmation, value types, or
   artifact root drift from capture.
6. Validate the generated plan artifact after the no-WING smoke command and
   expose it in `summary.json`/`summary.md`.

## Test plan

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_readonly_preflight_evidence_snapshot tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-wing-plan-preview --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

The main risk is wording: an offline plan could be mistaken for physical proof.
The implementation keeps `contacts_backend=false`, `queries_console=false`,
`execution_allowed_by_plan=false`, and `live_authority_granted=false`, and the
no-WING summaries still require real WING artifacts for readiness. Rollback is
to remove the plan-only flags, preview command, tests, runbook section, ADR, and
report. No live WING write path is affected.
