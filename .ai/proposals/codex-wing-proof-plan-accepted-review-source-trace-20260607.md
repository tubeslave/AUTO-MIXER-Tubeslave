# Codex Proposal: WING Proof Plan Accepted Review Source Trace

## Short Thesis

The supervised WING proof plan should name the accepted read-only preflight
source that authorizes planning the next physical proof.

## What I Understood

The system is still blocked on real WING evidence. Offline tooling should not
pretend to close that blocker, but it can make the operator handoff more
auditable by preserving the accepted `summary.json` source id in the next proof
plan.

## Proposed Solution

Copy `accepted_source_ids` and `rejected_source_ids` from the
`wing_readonly_preflight_review_decision_ledger` summary into the
`supervised_wing_proof_plan` summary. Extend the no-WING validator to compare
those fields against the ledger overlay and render them in generated summaries.

## Likely Files To Touch

- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `Docs/reports/AUTOMIXER_WING_PROOF_PLAN_ACCEPTED_REVIEW_SOURCE_TRACE_20260607.md`

## Alternatives Considered

- Keep only accepted counts. This is less actionable during the hardware
  handoff because the operator has to inspect a separate ledger to confirm the
  source.
- Copy full decision records into the proof plan. This is too noisy and risks
  making the plan a second ledger.

## Risks

The proof-plan summary grows by two fields. It remains read-only and
non-authoritative for live execution.

## Test Plan

- `python3 -m py_compile backend/supervised_wing_proof_plan.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_proof_plan_accepted_source_trace_20260607`

## Where The Other Agent May Disagree

Kimi may argue the ledger already contains this identity. The stronger handoff
is to include the accepted source in the exact proof plan the operator will use
before any physical fader/gain run.
