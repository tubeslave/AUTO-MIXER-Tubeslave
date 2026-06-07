# ADR: WING Read-Only Preflight Session-Ready Review Checklist - 2026-06-07

## Context

The WING read-only preflight runner requires operator confirmation that the
intended backend/WING session is ready before physical readback can start. The
captured summary records that confirmation, and evidence validation rejects
missing fields.

The manual review decision ledger is the next gate before supervised fader/gain
proof planning. It must prove that the operator reviewed the same safety facts
that the artifact validator requires.

## Options considered

1. Leave the checklist unchanged and rely on evidence validation.
2. Add prose to the runbook only.
3. Add `confirm_operator_session_ready` to the canonical required checklist ids.

## Decision

Use option 3. Accepted WING read-only preflight review decision records now must
include:

- `confirm_artifact_paths`
- `confirm_operator_channel_scope`
- `confirm_operator_session_ready`
- `confirm_physical_readbacks`
- `confirm_event_log_readonly_sequence`
- `confirm_no_live_authority`

The artifact review output also exposes
`operator_session_confirmation_required` and
`operator_confirmed_session_ready` so the operator can inspect the underlying
evidence before recording the decision.

## Why this won

The MVP gate is file-backed and fail-closed. A canonical checklist id is
machine-checkable, appears in pending decision templates, and prevents stale
review decisions from silently advancing proof planning.

## Rejected alternatives

Evidence validation alone proves the field exists, not that the manual review
decision acknowledged it. Runbook prose is useful but not enforceable.

## Implementation plan

1. Add the checklist id to the canonical decision ledger.
2. Add the same id to the review tool, handoff packet, no-WING release gate,
   and runbook.
3. Surface session-ready fields in artifact evidence rows and review markdown.
4. Update focused tests and documentation.

## Test plan

- Focused pytest for WING preflight evidence, review tools, decision ledger,
  handoff packet, and no-WING release checks.
- Quick no-WING release check.
- Direction guard and full pytest before commit.

## Risks and rollback

Risk: old pending decision templates fail with
`review_decision_checklist_ids_mismatch`. This is intended; the operator should
regenerate the template from the updated review tool.

Rollback is a small revert of the checklist id, artifact-row fields, tests, and
docs. No mixer state is touched.
