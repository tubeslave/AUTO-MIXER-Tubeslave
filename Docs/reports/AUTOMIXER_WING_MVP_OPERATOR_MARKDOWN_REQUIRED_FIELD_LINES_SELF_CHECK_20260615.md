# WING MVP Operator Markdown Required Field Lines Self-Check

Date: 2026-06-15
Status: implemented and validated

## Summary

The no-WING operator Markdown validator now fails closed when
`operator_markdown_summary_required_field_lines` is missing, malformed, or does
not list every field in the current operator Markdown contract.

The generated summary exposes:

- `operator_markdown_summary_required_field_lines_missing_members`
- `operator_markdown_summary_required_field_lines_match_contract`

## Safety Boundary

This is a local no-WING validation change. It does not contact WING, query a
console, start a backend/WING session, or grant live authority.

## Validation

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
  - `333 passed, 8 skipped`
- `PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend`
  - `status: passed`
  - `operator_markdown_summary_required_field_lines_missing_members: []`
  - `operator_markdown_summary_required_field_lines_match_contract: true`
  - `operator_markdown_summary_field_lines_rendered: true`
  - `operator_markdown_summary_missing_field_lines: []`
  - `live_authority_granted: false`
  - `full_autonomous_ready: false`
- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_supervised_wing_proof_plan.py`
  - `360 passed, 8 skipped`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - `3416 passed, 9 skipped`
- `PYTHONPATH=backend python tools/memory/memory_lint.py`
  - PASS
- `/opt/homebrew/bin/git diff --check`
  - PASS
