# Proposal: WING MVP no-WING prestart summary contract

## 1. Short thesis

Make `tools/no_wing_release_check.py` consume the shared WING read-only
preflight prestart contract so the gate that produces the no-WING summary
cannot drift from the tools that consume it.

## 2. What I understood

The first open blocker remains WING read-only connection proof. The no-WING
quick gate is not live proof, but its passed summary is required before the
real backend/WING session and later accepted-candidate trace.

## 3. Proposed solution

- Add a shared `WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_STATUS` constant.
- Keep backend check status tied to that expected passed status.
- Use shared summary type/status/policy and backend check name/status in:
  - `tools/wing_readonly_preflight_runner.py`
  - `backend/wing_readonly_preflight_evidence.py`
  - `tools/no_wing_release_check.py`
- Add tests proving no-WING summary writing and plan-artifact validation use
  the shared contract aliases.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## 5. Alternatives considered

Leaving no-WING literals in place would preserve current behavior but keep the
release gate as an independent contract source. That is weaker than a single
shared source.

## 6. Risks

The change is mechanical, but a bad alias could block valid plans. The rollback
is to revert no-WING call sites to the prior literals. No live authority is
introduced.

## 7. Test plan

- `py_compile` on touched modules.
- Focused tests for no-WING, runner, evidence, review, and proof plan.
- no-WING quick with isolated artifact root.
- Direction guard, Project Director, and full pytest.

## 8. Where the other agent may disagree

Kimi may argue the constants are simple enough to duplicate. The counterpoint
is that this is a safety gate, and the cost of a single contract source is low.
