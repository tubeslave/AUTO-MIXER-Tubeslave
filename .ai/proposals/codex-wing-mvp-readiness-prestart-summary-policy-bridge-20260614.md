# Codex Proposal: WING MVP Readiness Prestart Summary Policy Bridge

## Short thesis

Make the autonomy promotion readiness snapshot expose the same no-WING
prestart summary type and connection-policy contract that gates accepted
read-only candidates.

## What I understood

The active blocker remains the WING read-only connection proof. Review tooling
and proof planning already require the accepted-candidate prestart trace to bind
to a passed `no_wing_release_check` summary with
`no_hardware_no_network_no_live_writes`. The readiness bridge still surfaced
path, SHA, status, and no-contact flags without explicitly carrying the type
and policy match booleans.

## Proposed solution

Add `prestart_backend_port_summary_type`,
`prestart_backend_port_summary_type_matches_contract`,
`prestart_backend_port_summary_connection_policy`, and
`prestart_backend_port_summary_connection_policy_matches_contract` to the WING
read-only preflight evidence summary inside
`backend/autonomy_promotion_readiness.py`. Assert those values in
`tests/test_autonomy_promotion_readiness.py`.

## Likely files to touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `Docs/reports/AUTOMIXER_WING_MVP_READINESS_PRESTART_SUMMARY_POLICY_BRIDGE_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Keep the type and policy only in review/proof-plan validation. Rejected because
readiness review is the bridge operators inspect before matrix changes, and it
should display the full accepted-candidate prestart contract.

Make readiness closure depend on these fields immediately. Rejected because the
current MVP still forbids readiness closure from candidate/offline evidence;
the bridge is report-only until a real WING artifact is manually reviewed.

## Risks

The risk is confusing this as readiness closure. The fields remain report-only:
`readiness_closure_allowed_by_evidence`, `live_authority_granted`, and
`full_autonomous_ready` stay false.

## Test plan

Run focused readiness tests, focused WING/readiness rails, regenerate latest MVP
progress, run Project Director and direction guard, run `py_compile`, memory
lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer deriving all readiness fields from a typed contract object. I
agree long-term, but this keeps the bridge aligned with the existing contract
without broad refactoring.
