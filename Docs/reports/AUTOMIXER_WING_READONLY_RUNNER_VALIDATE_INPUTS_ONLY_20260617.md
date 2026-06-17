# Automixer WING Read-Only Runner Validate Inputs Only

## Summary

`tools/wing_readonly_preflight_runner.py` now supports
`--validate-run-inputs-only`. The mode validates the exact real-run arguments,
prestart summary, and expected plan, then exits before opening WebSocket.

## Safety Boundary

- `validation_only=true`
- `websocket_opened=false`
- `sent_request_types=[]`
- `contacts_backend=false`
- `queries_console=false`
- `live_authority_granted=false`
- No readiness closure is allowed.

## Validation

Current validation on 2026-06-17:

```bash
PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_runner.py
# PASS

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py
# 43 passed in 0.35s

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_codex_direction_guard.py
# 458 passed in 4.49s

rm -rf /tmp/automixer_validate_run_inputs_only
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_validate_run_inputs_only/no_wing
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/wing_readonly_preflight_runner.py --url ws://127.0.0.1:8765 --channels 3 --value-types fader,gain,pan --require-physical-readback --operator-confirm-channels 3 --operator-confirm-session-ready --prestart-summary /tmp/automixer_validate_run_inputs_only/no_wing/20260617T104003Z_no_wing_release_check/backend_port_prestart/summary.json --artifact-root /tmp/automixer_validate_run_inputs_only/wing_readonly_preflight --plan-only --plan-output /tmp/automixer_validate_run_inputs_only/expected-plan.json
PYTHONPATH=backend /opt/homebrew/bin/python3 tools/wing_readonly_preflight_runner.py --url ws://127.0.0.1:8765 --channels 3 --value-types fader,gain,pan --require-physical-readback --operator-confirm-channels 3 --operator-confirm-session-ready --prestart-summary /tmp/automixer_validate_run_inputs_only/no_wing/20260617T104003Z_no_wing_release_check/backend_port_prestart/summary.json --artifact-root /tmp/automixer_validate_run_inputs_only/wing_readonly_preflight --expected-plan /tmp/automixer_validate_run_inputs_only/expected-plan.json --validate-run-inputs-only
# status='ready_for_websocket'
# reason='run_inputs_validated_no_websocket'
# selected_mvp_blocker='WING read-only connection proof'
# selected_mvp_blocker_status='tooling_ready_waits_real_wing'
# validation_only=True
# websocket_opened=False
# expected_plan_match=True
# expected_plan_violations=[]
# contacts_backend=False
# queries_console=False
# execution_allowed_by_validation=False
# live_authority_granted=False
# sent_request_types=[]
# summary: /tmp/automixer_validate_run_inputs_only/wing_readonly_preflight/20260617T104010Z_wing_readonly_preflight/summary.json

PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py
# Status: PASS

PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q
# 3761 passed, 1 skipped, 1 warning in 47.04s

/opt/homebrew/bin/git diff --check -- tools/wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_runner.py .ai/briefs/wing-readonly-runner-validate-inputs-only-20260617.md .ai/proposals/codex-wing-readonly-runner-validate-inputs-only-20260617.md Docs/adr/wing-readonly-runner-validate-inputs-only-20260617.md Docs/reports/AUTOMIXER_WING_READONLY_RUNNER_VALIDATE_INPUTS_ONLY_20260617.md
# PASS
```

This is not a WING proof artifact. It is a final no-WebSocket readiness check
for the operator before capturing the real `wing_readonly_preflight.v1`.
