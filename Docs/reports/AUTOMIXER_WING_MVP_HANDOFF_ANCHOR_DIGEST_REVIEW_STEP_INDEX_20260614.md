# WING MVP Handoff Anchor Digest Review Step Index

## Summary

The no-WING handoff gate now treats the anchor digest review operator step index
as part of the shared WING read-only preflight contract.

Before supervised proof planning, the no-WING handoff summary must show:

- `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`
- `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`
- `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`

## Safety Boundary

This remains offline and read-only. It does not contact WING, execute packet
commands, grant live authority, or close the WING read-only connection proof
blocker.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py --tb=short`: 257 passed.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`: 268 passed.
- `python3 -m py_compile backend/wing_readonly_preflight_contract.py tools/no_wing_release_check.py tools/build_wing_readonly_preflight_handoff_packet.py tools/codex_autoloop/project_director_agent.py`: passed.
- `git diff --check`: passed.
- Default `tools/no_wing_release_check.py --quick` failed closed on an existing local `127.0.0.1:8765` listener and generated an operator confirmation template without live authority.
- Free-port no-WING quick validation passed with `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`, `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`, `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, supervised gate index match true, and no handoff violations.
- Project Director: `on_target`, score `90`, selected blocker `WING read-only connection proof`, selected evidence file count `62`, new report present.
- Direction guard: `on_target`, score `90`, `stop_the_line: false`.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`: PASS.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`: 3295 passed, 1 skipped, 1 warning.
