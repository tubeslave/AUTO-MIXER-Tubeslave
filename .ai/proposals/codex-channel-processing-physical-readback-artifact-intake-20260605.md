# Codex Proposal: Channel Processing Physical Readback Artifact Intake

## Short Thesis

Add a transport-free artifact intake for channel HPF/EQ/compressor/gate physical readbacks, then surface it through release evidence and Product Layer without changing live authority.

## What I Understood

The channel processing contract defines what future physical readback evidence must contain for channel HPF, EQ, compressor, and gate state. The missing layer is a safe intake that can find operator-captured JSON artifacts, validate them against that contract, and keep them as manual-review candidates.

## Proposed Solution

- Add `backend/channel_processing_physical_readback_artifact_intake.py`.
- Register it as `shadow_only` analysis in `processing_registry`.
- Include it in `console_read_parameters` capability support paths.
- Validate it from `tools/no_wing_release_check.py` after the channel processing contract.
- Expose normalized read-only status in Product Layer and cockpit module status.
- Add tests that reject unsafe claims and keep fixture closure blocked.

## Likely Files To Touch

- `backend/channel_processing_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Relevant backend/frontend tests

## Alternatives Considered

- Reuse channel basic-controls artifact intake: rejected because channel processing needs HPF/EQ/dynamics-specific channel validation and promotion-family reporting.
- Treat a valid artifact as fixture closure: rejected because operator review and physical proof promotion policy are separate gates.

## Risks

- A valid candidate could be mistaken for live readiness.
- Malformed summary artifacts could claim live authority or fixture closure.
- Frontend could display intake status as an action surface.

## Test Plan

- Validate safe defaults and transport-free source.
- Validate valid artifacts remain manual-review candidates.
- Validate no-WING rejects authority claims.
- Validate Product Layer normalizes unsafe claims to false.
- Validate frontend rows remain `canWriteLive=false`.

## Where The Other Agent May Disagree

Kimi may argue this should wait until channel basic-control artifact intake is exercised with real operator captures. The counterpoint is that EQ/dynamics/gate proof is an independent prerequisite for EQ, compression, gate, de-essing, frequency separation, and unmasking promotion, and landing intake as read-only does not widen operational authority.
