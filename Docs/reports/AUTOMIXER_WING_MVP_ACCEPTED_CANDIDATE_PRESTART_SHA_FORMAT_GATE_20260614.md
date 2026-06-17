# Automixer WING MVP Accepted-Candidate Prestart SHA Format Gate

Date: 2026-06-14
Status: Validated

## Summary

Accepted-candidate WING read-only prestart traces now reject malformed
`backend_port_prestart_summary_sha256` values. The field must be a lowercase
64-character SHA-256 hex digest before an accepted review source can be treated
as prestart-contract-valid for supervised proof planning.

## Changed Surfaces

- `backend/wing_readonly_preflight_contract.py` now distinguishes missing
  no-WING prestart summary digests from malformed digest strings.
- `tests/test_wing_readonly_preflight_review_decision.py` covers the shared
  helper and review-decision ledger fail-closed behavior for malformed
  prestart digests.
- `tests/test_supervised_wing_proof_plan.py` covers malformed digest blocking
  of physical proof-plan steps.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Validation

Refreshed on 2026-06-17:

- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile backend/wing_readonly_preflight_contract.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py`
  returned `439 passed in 4.23s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
  returned `520 passed in 4.48s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_prestart_sha_format_gate`
  generated
  `/tmp/automixer_prestart_sha_format_gate/20260617T094419Z_no_wing_release_check`
  with `status: passed`, `failed_checks: []`,
  `readiness_closure_allowed_by_packet: false`,
  `live_authority_granted: false`, and `full_autonomous_ready: false`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/verify_no_wing_artifact_manifest.py /tmp/automixer_prestart_sha_format_gate/20260617T094419Z_no_wing_release_check`
  reported `status: passed`, `failed_checks: []`,
  `artifact_manifest_contract_cross_surface_match: true`,
  `artifact_manifest_self_consistency_match: true`, and
  `artifact_manifest_self_consistency_summary_surface_match: true`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
  returned `3758 passed, 1 skipped, 1 warning in 48.42s`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`.
- PASS: `/opt/homebrew/bin/git diff --check -- .ai/briefs/wing-mvp-accepted-candidate-prestart-sha-format-gate-20260614.md .ai/proposals/codex-wing-mvp-accepted-candidate-prestart-sha-format-gate-20260614.md Docs/adr/wing-mvp-accepted-candidate-prestart-sha-format-gate-20260614.md Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_PRESTART_SHA_FORMAT_GATE_20260614.md`.
- PASS: `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
  selected `WING read-only connection proof`, reported
  `guard_status: drifting`, `guard_score: 30`,
  `selected_evidence_file_count: 205`, and kept current-command baseline
  coverage passing with no missing markers.
