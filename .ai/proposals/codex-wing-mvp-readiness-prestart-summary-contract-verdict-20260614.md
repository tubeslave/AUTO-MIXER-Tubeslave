# Codex Proposal: WING MVP Readiness Prestart Summary Contract Verdict

## Short thesis

Surface a single prestart summary contract verdict and violation list in the
autonomy promotion readiness bridge.

## What I understood

The active blocker remains the WING read-only connection proof. The readiness
bridge now exposes the prestart summary type and connection policy, but manual
review still has to infer whether the whole accepted-candidate prestart
contract is valid. The shared contract validator already exists and should be
used instead of duplicating checks.

## Proposed solution

Import `wing_readonly_preflight_candidate_prestart_contract_violations` into
`backend/autonomy_promotion_readiness.py`. Emit
`prestart_backend_port_summary_contract_valid` and
`prestart_backend_port_summary_contract_violations` in the read-only preflight
readiness overlay. When no valid read-only candidate exists, emit a fail-closed
`no_valid_readonly_preflight_candidate` violation.

## Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `Docs/reports/AUTOMIXER_WING_MVP_READINESS_PRESTART_SUMMARY_CONTRACT_VERDICT_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Keep only individual field matches. Rejected because a reviewer still needs an
aggregate verdict before considering readiness rows.

Duplicate the validation logic locally in readiness. Rejected because the
shared WING prestart contract already owns the authoritative checks.

## Risks

The main risk is treating the verdict as readiness closure. The snapshot still
keeps `readiness_closure_allowed_by_evidence`, `live_authority_granted`, and
`full_autonomous_ready` false.

## Test plan

Run focused readiness tests, focused WING/readiness rails, regenerate latest MVP
progress, run Project Director and direction guard, run `py_compile`, memory
lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer moving all readiness overlay construction into a typed WING
contract adapter. I agree that is cleaner later; this change keeps the first
blocker moving with the existing shared validator.
