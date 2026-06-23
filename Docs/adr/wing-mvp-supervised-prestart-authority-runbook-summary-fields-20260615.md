# ADR: WING MVP Supervised Prestart Authority Runbook Summary Fields

## Context

The supervised WING proof path must remain behind manual acceptance and must not
grant live authority from offline no-WING artifacts. The structured
`supervised_prestart_gate` records this, but release summaries should make the
authority denial and runbook target directly visible.

## Options Considered

1. Keep authority/runbook details only inside the structured gate.
2. Mirror authority/runbook details as top-level no-WING summary fields.
3. Replace the structured gate with flat fields only.

## Decision

Mirror `grants_live_authority`, `requires_operator_decision_status`, and
`unlocks_runbook` as top-level derived summary fields while preserving the
structured gate.

## Why This Won

The selected option improves operator review and automation checks without
changing runtime authority or WING communication behavior.

## Rejected Alternatives

- Nested-only validation is less reviewable.
- Flat-only validation loses structured handoff context.

## Implementation Plan

- Derive fields from `supervised_prestart_gate`.
- Render them in no-WING JSON and Markdown summaries.
- Add focused assertions for artifact, top-level summary, and Markdown parity.
- Add baseline markers and documentation.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the derived fields and marker docs. The structured gate
remains source of truth.
