# Automixer WING MVP Handoff Required-False Detail Contract

Date: 2026-06-14
Status: Validated

## Summary

The no-WING release gate now keeps the backend-port required-false detail fields
bound through the offline WING read-only plan, handoff packet, JSON summary, and
Markdown summary.

## Why

The first WING blocker remains `WING read-only connection proof`. Offline
evidence must prepare that run without implying live authority. The required
backend-port no-authority fields are safety-critical enough that they must not
be lost between prestart summary, plan, packet, and operator-visible report.

## Changed surfaces

- `tools/no_wing_release_check.py` validates the shared required-false field set
  in plan artifacts and compares handoff packet values against plan values.
- `tools/build_wing_readonly_preflight_handoff_packet.py` emits those fields in
  the handoff packet and packet Markdown.
- `tests/test_no_wing_release_check.py` covers valid, missing, unsafe, and
  packet/plan drift cases.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

- Focused no-WING/rails/Project Director pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py tests/test_wing_mvp_rails.py
  tests/test_project_director_agent.py --tb=short -x`: 277 passed,
  8 skipped in 3.19s.
- Targeted plan/handoff regression pytest:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q
  tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_plan_artifact
  tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_wing_preflight_plan_missing_required_backend_false_detail_fields
  tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_plan_artifact
  tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact
  tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_handoff_packet_with_prestart_summary_contract_drift
  tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_plan_artifact
  tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_wing_preflight_handoff_packet_artifact
  --tb=short -x`: 7 passed in 0.96s.
- no-WING quick without frontend:
  `PYTHONPATH=backend /opt/homebrew/bin/python3
  tools/no_wing_release_check.py --quick --skip-frontend --artifact-root
  /tmp/automixer_goal_required_false_detail_nowing`: passed with
  `connection_policy: no_hardware_no_network_no_live_writes`,
  `live_write: false`, `failed_checks: []`, required-false plan fields all
  `false`, and required-false handoff match flags all `true`.
- Project Director JSON selected `WING read-only connection proof`,
  `selected_blocker_status: tooling_ready_waits_real_wing`,
  `selected_action_source: Current First Command`, `selected_evidence_file_count:
  102`, `current_first_command_baseline_coverage_passed: true`,
  `guard_status: on_target`, `guard_score: 90`, and `guard_stop_the_line:
  false`.
- Direction guard JSON reported `status: on_target`, `score: 90`, and
  `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile
  tools/no_wing_release_check.py
  tools/build_wing_readonly_preflight_handoff_packet.py
  tools/codex_autoloop/project_director_agent.py
  tools/codex_autoloop/update_mvp_progress.py`: passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`:
  PASS.
- `/opt/homebrew/bin/git diff --check`: passed.
- Standard full suite:
  `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x
  --tb=short -q`: 3307 passed, 9 skipped, 1 warning in 71.28s.
