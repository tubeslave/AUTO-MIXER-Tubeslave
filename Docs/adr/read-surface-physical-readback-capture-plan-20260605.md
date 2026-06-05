# ADR: Read Surface Physical Readback Capture Plan

## Context

Read-surface fixture evidence now exposes the first remaining physical readback
gap. The remediation plan turns that gap into a planning action, but the project
also needs an operator-facing checklist that names what must be captured before
future proof artifacts can be trusted.

This repository controls live-sound behavior. A checklist must not become a
hardware runner, a console query path, or a promotion signal.

## Options considered

1. Keep only fixture and remediation summaries.
2. Add a non-executing capture-plan read model.
3. Add a hardware capture runner.

## Decision

Use a non-executing capture-plan read model.

The capture plan is shadow-only planning evidence. It can name the next physical
readback surface, matching remediation action, required artifact contracts,
prerequisites, abort conditions, and forbidden actions. It must not query
hardware, execute tests, mutate console state, or grant live authority.

## Why this won

This gives the operator a concrete next proof target while preserving the
supervised safety boundary. It also lets no-WING release summaries and frontend
status rows display the next hardware artifact requirement without implying that
Codex, Paperclip, Product Layer, or the UI may run the capture.

## Rejected alternatives

Keeping only summary metadata was too vague for the next hardware-proof step.
It exposed the target but not the capture contract, abort conditions, or
forbidden actions.

Adding a hardware capture runner was rejected for this slice. It would need a
separate operator-approved design for connection scope, readback storage,
transport hardening, hardware identity, and artifact review.

Treating cached state as physical proof was rejected. Cached DTOs and no-WING
fixtures are useful planning evidence, but they do not close physical readback
fixture gaps.

## Implementation plan

- Build capture-plan metadata from fixture evidence and remediation plan data.
- Register the method as shadow-only and non-executor in the processing
  registry.
- Surface the plan through no-WING summaries, Product Layer proof quality, and
  frontend read-only module rows.
- Serialize `live_write=false`, `execution_allowed_by_plan=false`,
  `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Keep the module transport-free and test that forbidden live/apply tokens stay
  absent.

## Test plan

- Focused capture-plan tests for target/action consistency and non-execution
  safety flags.
- no-WING release-check tests for schema alignment, corrupted execution claims,
  JSON summary, and Markdown summary.
- Product Layer tests for normalization of source claims back to read-only
  evidence.
- Frontend adapter contract tests for read-only status display.
- Full backend suite, no-WING quick check, memory lint, and diff hygiene before
  relying on the slice.

## Risks and rollback

The main risk is a later agent or operator treating the checklist as authority
to run hardware operations. Rollback is straightforward: remove the capture-plan
method from registry/Product Layer/frontend/no-WING surfaces and keep the
earlier fixture/remediation summaries. No mixer state is changed by this ADR or
the capture-plan code.
