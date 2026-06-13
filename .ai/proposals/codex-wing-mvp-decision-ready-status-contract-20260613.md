# Codex Proposal: WING MVP decision-ready status contract

## 1. Short thesis

Make `accepted_decision_ready` a shared WING preflight contract value for both
the decision-review CLI and the supervised prestart gate.

## 2. What I understood

The real blocker is still the first supervised WING MVP row: read-only
connection proof on real hardware. Current offline work must only reduce the
chance of an unsafe or stale handoff, not claim readiness.

## 3. Proposed solution

- Import
  `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS`
  into `tools/review_wing_readonly_preflight_decisions.py`.
- Expose a local alias for CLI readability.
- Use the alias in the decision-review exit-code table, report status, and
  exit-code selection.
- Add tests proving the CLI ready status matches
  `expected_supervised_prestart_gate()["requires_operator_decision_status"]`.

## 4. Likely files to touch

- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `Docs/adr/wing-mvp-decision-ready-status-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_DECISION_READY_STATUS_CONTRACT_20260613.md`

## 5. Alternatives considered

- Leave the local string in the CLI. Rejected because the supervised gate and
  CLI can drift silently.
- Move all decision-review statuses into the shared contract. Rejected for this
  step because only the accepted-ready status unlocks the next supervised gate.

## 6. Risks

- Low implementation risk; the change is a constant binding.
- Test risk is mostly around status literals in existing tests.

## 7. Test plan

- Run focused tests for decision review, proof plan, no-WING release check, and
  WING rails.
- Run the standard full pytest suite before commit.

## 8. Where the other agent may disagree

Kimi might prefer centralizing every decision-review status now. The smaller
change is safer because only `accepted_decision_ready` is part of the supervised
proof gate contract.
