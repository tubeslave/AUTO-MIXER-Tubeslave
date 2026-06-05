# Codex Proposal: Routing/Sends Physical Readback Artifact Intake

## Short thesis

Add an artifact intake that validates operator-captured routing/sends readback
JSON without granting fixture closure or live authority.

## What I understood

The routing/sends schema contract exists, but future physical readback artifacts
need a safe intake path. Static route, bus send, main send, matrix send, and
output-route evidence are safety-sensitive because accepting stale or cached
data as proof could lead to unsafe routing automation later.

## Proposed solution

- Add `backend/routing_sends_physical_readback_artifact_intake.py`.
- Reuse the existing routing/sends contract validator.
- Support explicit paths, inline payloads, and default artifact-root discovery.
- Mark valid artifacts as manual-review candidates only.
- Add release-gate, Product Layer, and frontend rows that normalize all live,
  execution, closure, and full-auto claims to false.

## Likely files to touch

- `backend/routing_sends_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Tests for the new intake, release gate, Product Layer, and frontend adapter
- `.ai/briefs/`, `.ai/proposals/`, `.ai/memory/project.md`, `Docs/adr/`

## Alternatives considered

1. Treat a valid artifact as fixture closure.
2. Extend the routing/sends contract only and defer intake.
3. Add a hardware reader.

## Risks

The main risk is that a valid routing artifact is mistaken for route/send write
authority. The mitigation is consistent summary and UI wording:
`physical_fixture_gap_remains=true`, `fixture_closure_allowed_by_intake=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

## Test plan

- Focused backend intake, registry, capability, release-gate, and Product Layer
  tests.
- Frontend adapter contract test.
- no-WING quick release check.
- Full backend pytest before commit.

## Where the other agent may disagree

Kimi may prefer to wait for a hardware capture runner. I think the safer step is
to land the intake contract now because it makes future operator artifacts
reviewable without widening live authority.
