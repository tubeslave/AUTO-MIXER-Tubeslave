# WING MVP Handoff Declared Anchor Digest Review Index

## Summary

The no-WING handoff gate now validates the packet-declared
`mvp_blocker_anchor_digest_review_step_index` against both the shared contract
and the actual operator sequence.

Before supervised proof planning, the no-WING handoff summary must show:

- `mvp_blocker_anchor_digest_review_step_index: 8`
- `expected_mvp_blocker_anchor_digest_review_step_index: 8`
- `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`
- `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`

## Safety Boundary

This remains offline and read-only. It does not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short`: 258 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`: 269 passed.
- `python3 -m py_compile tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/project_director_agent.py backend/wing_readonly_preflight_contract.py`: passed.
- `git diff --check`: passed.
- Free-port no-WING quick validation passed with `mvp_blocker_anchor_digest_review_step_index: 8`, `expected_mvp_blocker_anchor_digest_review_step_index: 8`, `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`, and no handoff violations.
- Project Director: `on_target`, score `90`, selected blocker `WING read-only connection proof`, selected evidence file count `63`, new report present.
- Direction guard: `on_target`, score `90`, `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3296 passed, 1 skipped, 1 warning.
