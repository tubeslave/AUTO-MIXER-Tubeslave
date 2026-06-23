# ADR: WING MVP Supervised Prestart Marker Verdict Summary Fields

## Context

The supervised proof path depends on a specific final gate marker list:
`accepted_decision_ready`, accepted-candidate prestart trace, artifact-root
trace, expected-artifact source binding, and the supervised fader/gain proof.
The list was visible, but no top-level verdict proved the structured gate's
markers matched the contract and summary list.

## Options Considered

1. Keep the marker list without verdict fields.
2. Add top-level derived marker match verdict fields.
3. Replace structured gate markers with only top-level fields.

## Decision

Add top-level derived fields for the structured gate marker list and marker
match verdicts.

## Why This Won

It makes operator review and automation checks direct while preserving the
structured gate as source of truth. It does not widen live authority or contact
WING hardware.

## Rejected Alternatives

- List-only validation is harder to audit.
- Flat-only validation loses structured handoff context.

## Implementation Plan

- Derive `supervised_prestart_gate_required_markers`.
- Compare it to the shared marker contract and top-level summary marker list.
- Render JSON/Markdown fields.
- Add tests and Project Director baseline markers.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the derived fields and marker docs. The structured gate
and existing marker list remain intact.
