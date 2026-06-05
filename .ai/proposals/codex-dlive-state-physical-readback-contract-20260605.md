# Codex Proposal: dLive State Physical Readback Contract

## Short thesis

Add `dlive_state_physical_readback_contract` as a transport-free schema gate for
future dLive physical readback and WING parity artifacts, without adding MIDI/TCP
sends or live authority.

## What I understood

The existing dLive DTO is useful cached visibility, but it is not enough proof
for autonomous or supervised promotion decisions. dLive parity must stay blocked
until a separately captured physical artifact proves channel identity, fader
scale, mute state, raw MIDI state, and WING mapping.

## Proposed solution

Create a backend contract module with constants, a contract builder, and a
validator for future physical artifacts. Register it as a `shadow_only`
evaluation method, surface it through console-read capability tracking,
fixture/remediation evidence, no-WING summaries, Product Layer proof quality,
and frontend Module Status rows.

## Likely files to touch

- `backend/dlive_state_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Related backend/frontend tests and council docs.

## Alternatives considered

Keep dLive covered by cached DTO only. Rejected because cache cannot prove
physical state or WING parity. Add transport query/apply behavior now. Rejected
because it widens the live surface before readback, rollback, freshness, and
operator proof exist.

## Risks

The schema may be misread as hardware readiness. Mitigation: keep all summaries
read-only with `physical_fixture_gap_remains=true`,
`accepted_as_physical_fixture_candidate=false`, `live_authority_granted=false`,
and explicit `midi_send_allowed=false`, `tcp_send_allowed=false`,
`dlive_write_allowed=false`.

## Test plan

Run the new contract tests, registry/capability/fixture/remediation/no-WING
tests, Product Layer tests, frontend adapter tests, no-WING quick validation,
memory lint, frontend build, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer waiting for a real dLive physical artifact. The counterpoint is
that the schema gate must exist first so the future artifact has a stable
acceptance contract and cannot silently claim write authority.
