# ADR: WING MVP Supervised Prestart Trace Summary Fields

## Context

Supervised WING fader/gain proof planning must remain blocked until the
accepted read-only candidate has valid prestart, artifact-root, and
expected-artifact traces. The structured `supervised_prestart_gate` records
these requirements, but flat summary surfaces make operator review and
automation checks more direct.

## Options Considered

1. Keep the trace prerequisites only inside `supervised_prestart_gate`.
2. Mirror all trace prerequisites as top-level no-WING summary fields.
3. Replace the structured gate with flat fields only.

## Decision

Mirror the accepted-candidate prestart trace and artifact-root trace
requirements as top-level derived fields while preserving the structured gate
and existing expected-artifact summary field.

## Why This Won

The chosen approach keeps the structured contract as source of truth and adds a
small operator-readable surface. It does not contact WING hardware, grant live
authority, or close readiness.

## Rejected Alternatives

- Nested-only validation makes the gate harder to audit from release summaries.
- Flat-only validation loses structured context required by handoff packet
  consumers.

## Implementation Plan

- Derive the two new fields from `supervised_prestart_gate`.
- Render them in no-WING JSON and Markdown summaries.
- Add tests for artifact, top-level summary, and Markdown parity.
- Add Project Director baseline markers and documentation.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest around no-WING, Project Director, handoff packet, supervised
  proof plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the derived fields, marker docs, and focused assertions.
The structured supervised prestart gate remains intact.
