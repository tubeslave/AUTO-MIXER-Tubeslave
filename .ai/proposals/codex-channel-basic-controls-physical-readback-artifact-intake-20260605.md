# Codex Proposal: Channel Basic Controls Physical Readback Artifact Intake

## Short Thesis

Add a transport-free artifact intake for channel identity/basic-control physical readbacks, then surface it through release evidence and Product Layer without changing live authority.

## What I Understood

The channel basic-controls contract defines what future physical readback evidence must contain. The missing layer is a safe intake that can find operator-captured JSON artifacts, validate them against that contract, and keep them as manual-review candidates.

## Proposed Solution

- Add `backend/channel_basic_controls_physical_readback_artifact_intake.py`.
- Register it as `shadow_only` analysis in `processing_registry`.
- Include it in `console_read_parameters` capability support paths.
- Validate it from `tools/no_wing_release_check.py` after the channel basic-controls contract.
- Expose normalized read-only status in Product Layer and cockpit module status.
- Add tests that reject unsafe claims and keep fixture closure blocked.

## Likely Files to Touch

- `backend/channel_basic_controls_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Relevant backend/frontend tests

## Alternatives Considered

- Reuse raw mixer-state artifact intake: rejected because channel basic controls have channel-count and promotion-family fields that should be explicit.
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

Kimi may argue this should wait until the raw mixer-state artifact intake is fully exercised with real operator captures. The counterpoint is that channel basic-controls proof is an independent fader/gain/mute/pan prerequisite, and landing intake as read-only does not widen operational authority.
