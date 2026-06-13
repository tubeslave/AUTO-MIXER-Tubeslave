# Codex Proposal: WING MVP evidence value-types contract

## 1. Short thesis

Read-only preflight evidence must accept the same `value_types` as the
runner/handoff contract, not whatever the broader live-write policy currently
allows.

## 2. What I understood

The current project rail is still the supervised WING Rack proof. The next real
step is a real WING read-only preflight, but offline tooling can still reduce
the chance that stale or widened artifacts are accepted for manual review.

## 3. Proposed solution

- Change `backend/wing_readonly_preflight_evidence.py` so `ALLOWED_VALUE_TYPES`
  is derived from `WING_READONLY_PREFLIGHT_RUNNER_SUPPORTED_VALUE_TYPES`.
- Extend `tools/no_wing_release_check.py` to report
  `evidence_allowed_value_types` and fail closed if they differ from the shared
  contract.
- Add focused tests for the normal contract match and a monkeypatched mismatch.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/wing-mvp-evidence-value-types-contract-20260613.md`
- `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_VALUE_TYPES_CONTRACT_20260613.md`

## 5. Alternatives considered

- Keep evidence tied to `CURRENT_PROVEN_LIVE_WRITE_KINDS`. Rejected because
  that policy can expand independently from the current read-only proof scope.
- Duplicate the tuple in evidence intake. Rejected because it preserves drift
  risk.

## 6. Risks

Risk is low. If future WING read-only preflight needs more inspected value
types, the shared contract must be updated intentionally and all consumers will
move together.

## 7. Test plan

- Compile changed Python files.
- Run focused evidence/no-WING tests.
- Run no-WING quick.
- Run WING rails and full pytest.

## 8. Where the other agent may disagree

Kimi may argue the broader product policy should define inspection scope. For
the current MVP, the safer source is the explicit WING read-only preflight
contract because it is what the handoff and runner actually expose.
