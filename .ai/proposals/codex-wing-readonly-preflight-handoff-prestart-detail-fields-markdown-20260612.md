# Proposal: WING Read-Only Preflight Handoff Prestart Detail Fields Markdown

## Short thesis

Render the prestart failure detail field list in the markdown handoff packet so
the human-facing packet matches the JSON contract.

## What I understood

The current WING MVP blocker needs a clean read-only hardware proof. Offline
handoff artifacts should make the operator inspect the same fail-closed busy-port
fields that validators already require.

## Proposed solution

- Add `backend_port_prestart_check_failure_detail_fields` to the markdown
  Safety Boundary.
- Keep the source of truth as the existing shared contract and packet field.
- Extend the handoff markdown test to assert every shared field is rendered.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `.ai/briefs/...`
- `.ai/proposals/...`
- `Docs/adr/...`
- `Docs/reports/...`

## Alternatives considered

- Leave the markdown packet unchanged because JSON already contains the field.
- Add custom prose instead of rendering the machine-readable list.

## Risks

The markdown Safety Boundary gains one line. This is acceptable because it
prevents operator-facing drift from the release-gated JSON contract.

## Test plan

- Focused handoff markdown test.
- Adjacent WING/no-WING stack.
- `git diff --check`, memory lint, and full backend suite.

## Where the other agent may disagree

Kimi may prefer prose in the operator sequence. A rendered field list is easier
to validate and keeps generated markdown aligned with generated JSON.
