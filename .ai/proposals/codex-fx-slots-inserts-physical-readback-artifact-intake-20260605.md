# Codex Proposal: FX Slots/Inserts Physical Readback Artifact Intake

## Short thesis

Add an artifact intake that validates operator-captured FX slots/inserts
readback JSON without granting fixture closure or live authority.

## What I understood

The FX slots/inserts schema contract now exists, but the project still needs a
safe way to ingest future physical readback artifacts. Intake must remain
operator-reviewed and read-only because an artifact can be malformed, stale, or
captured from the wrong surface.

## Proposed solution

- Add `backend/fx_slots_inserts_physical_readback_artifact_intake.py`.
- Reuse the existing FX slots/inserts contract validator.
- Support explicit paths, inline payloads, and default artifact-root discovery.
- Mark valid artifacts as manual-review candidates only.
- Add release-gate, Product Layer, and frontend rows that normalize all live,
  execution, closure, and full-auto claims to false.

## Likely files to touch

- `backend/fx_slots_inserts_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Tests for the new intake, release gate, Product Layer, and frontend adapter
- `.ai/briefs/`, `.ai/proposals/`, `.ai/memory/project.md`, `Docs/adr/`

## Alternatives considered

1. Treat a valid artifact as fixture closure.
2. Extend the FX contract only and defer intake.
3. Add a hardware reader.

## Risks

The main risk is that a valid JSON artifact is mistaken for hardware proof
closure. The mitigation is consistent summary and UI wording:
`physical_fixture_gap_remains=true`, `fixture_closure_allowed_by_intake=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

## Test plan

- Focused backend intake, registry, capability, release-gate, and Product Layer
  tests.
- Frontend adapter contract test.
- no-WING quick release check.
- Full backend pytest before commit.

## Where the other agent may disagree

Kimi may prefer to defer intake until a physical capture runner exists. I think
the safer step is to land the intake contract now because it defines exactly
what future operator artifacts may and may not prove.
