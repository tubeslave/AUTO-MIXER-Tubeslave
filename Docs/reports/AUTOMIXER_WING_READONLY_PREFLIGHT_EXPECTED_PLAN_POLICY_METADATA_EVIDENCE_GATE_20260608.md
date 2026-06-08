# Automixer WING Read-Only Preflight Expected Plan Policy Metadata Evidence Gate - 2026-06-08

## Summary

This slice strengthens the first supervised WING MVP blocker: WING read-only
connection proof. The read-only preflight artifact evaluator now rejects a real
`summary.json` candidate when its referenced expected-plan file has mismatched
policy metadata.

The cycle also fixed the WING read-only handoff packet builder so the exact
repo-local script command used by no-WING release checks runs successfully from
the repository root.

## Safety Boundary

- No WING Rack session was opened.
- No backend WebSocket was opened for a real WING session.
- No OSC or mixer write command was sent.
- No supervised fader/gain proof was attempted.
- No readiness row or MVP blocker was closed.

## Changed Files

- `backend/wing_readonly_preflight_evidence.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `Docs/adr/wing-readonly-preflight-expected-plan-policy-metadata-evidence-gate-20260608.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_POLICY_METADATA_EVIDENCE_GATE_20260608.md`

## Why This Moves The MVP

The next real action remains the read-only WING preflight. This change makes the
captured artifact review stricter: a linked expected-plan file must still point
to `WING read-only connection proof`, keep `operator_required=true`, keep the
allowed request list limited to `get_operator_mode`, `get_state`, and
`query_mixer_parameter`, include the full forbidden request list, and preserve
the required fail-closed conditions.

That reduces the chance that a stale or tampered plan file could be paired with
a real preflight summary and appear safe during manual review.

The handoff packet builder fix reduces the same blocker because the operator
handoff packet is part of the first real hardware step. The command must run as
`python3 tools/build_wing_readonly_preflight_handoff_packet.py ...`, not only as
an imported module.

## Validation

Run focused validation after this change:

```bash
python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py
PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py
PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle --command-timeout-sec 300
python3 tools/memory/memory_lint.py
```

## Remaining WING MVP Blocker

The first blocker remains `tooling_ready_waits_real_wing`. The operator still
must run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` against the intended
backend/WING session, preserve the real `summary.json` plus `events.jsonl`, and
record exactly one manual review decision before supervised fader/gain proof can
begin.
