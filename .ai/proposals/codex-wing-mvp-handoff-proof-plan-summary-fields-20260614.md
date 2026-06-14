# Codex Proposal: WING MVP Handoff Proof-Plan Summary Fields

## Short thesis

Make the WING read-only handoff packet explicitly require the supervised
proof-plan summary fields that prove accepted-candidate prestart diagnostics
before any supervised proof can be attempted.

## What I understood

The current blocker is still the real WING read-only connection proof. The
proof plan now computes and displays canonical expected prestart diagnostics,
but the operator-facing handoff packet should also say which proof-plan fields
must be reviewed before moving into `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Proposed solution

Add a shared contract constant for required supervised proof-plan summary
fields. Emit it in `tools/build_wing_readonly_preflight_handoff_packet.py`,
render it in Markdown, validate it in `tools/no_wing_release_check.py`, and
include it in no-WING JSON/Markdown summaries.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

Leave the handoff packet implicit. Rejected because the operator handoff is
the bridge between read-only proof and supervised proof; hidden summary-field
requirements are easy to miss.

Only update documentation. Rejected because a stale handoff packet would still
pass the release gate.

## Risks

The main risk is adding another static list to maintain. Keeping it in the
shared contract module and testing the builder plus no-WING gate limits drift.
This change does not grant execution authority.

## Test plan

Run focused handoff/no-WING tests, focused WING/readiness rail tests, Project
Director, direction guard, no-WING quick fail-closed check, `py_compile`,
memory lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer embedding full field semantics in the handoff packet rather
than field names. That can be added later; this patch first makes the required
summary surface explicit and gate-enforced.
