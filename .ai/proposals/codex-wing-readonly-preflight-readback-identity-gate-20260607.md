# Proposal: WING Read-Only Preflight Readback Identity Gate

## 1. Short thesis

The preflight runner should never emit a `passed` physical-readback artifact
unless every readback matches the exact channel, value type, and address that
was queried.

## 2. What I understood

The current MVP first blocker is still real WING read-only connection proof.
The runner already sends only read-only requests and the review layer catches
event-log identity mismatches, but the runner should fail closed earlier so a
bad artifact is clearly blocked before manual review.

## 3. Proposed solution

- Add a small `readback_identity_matches()` helper to
  `tools/wing_readonly_preflight_runner.py`.
- During each real `query_mixer_parameter`, add
  `readback_identity_mismatch:chN:<value_type>` to blockers when the response
  identity differs.
- Add `readback_identity_mismatch` to the offline plan fail-closed condition
  list and to the no-WING required condition list.
- Add a focused runner test for a physical readback on the wrong channel.

## 4. Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-readonly-preflight-readback-identity-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-readback-identity-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READBACK_IDENTITY_GATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Leave identity mismatch to artifact review only.
- Treat wrong-identity physical readback as a warning.
- Reject all `mixer_parameter_read_blocked` rows without preserving backend
  reason strings.

## 6. Risks

The main risk is false confidence. This change does not create real WING proof;
it only ensures a real run fails closed if the backend returns the wrong
readback identity.

## 7. Test plan

- py_compile for runner and no-WING release check.
- Focused runner and no-WING plan-artifact tests.
- no-WING quick run.
- memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may say artifact review already catches this. The runner-side gate is still
worth it because the artifact's own `status` should not say `passed` when its
readback identity is wrong.
