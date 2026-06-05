# Codex Proposal: Bus Group Main DCA Physical Readback Artifact Intake

## Short Thesis

Add a transport-free artifact intake for bus/group/main/DCA physical readbacks, then surface it through release evidence and Product Layer without changing live authority.

## What I Understood

The bus/group/main/DCA contract defines what future physical readback evidence must contain for bus controls, main controls, DCA controls, assignments, and metering. The missing layer is a safe intake that can find operator-captured JSON artifacts, validate them against that contract, and keep them as manual-review candidates.

## Proposed Solution

- Add `backend/bus_group_main_dca_physical_readback_artifact_intake.py`.
- Register it as `shadow_only` analysis in `processing_registry`.
- Include it in `console_read_parameters` capability support paths.
- Validate it from `tools/no_wing_release_check.py` after the bus/group/main/DCA contract.
- Expose normalized read-only status in Product Layer and cockpit module status.
- Add tests that reject unsafe claims and keep fixture closure blocked.

## Likely Files To Touch

- `backend/bus_group_main_dca_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Relevant backend/frontend tests

## Alternatives Considered

- Reuse channel artifact intake: rejected because bus/main/DCA proof needs bus, main, and DCA counts plus assignment and metering-specific reporting.
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

Kimi may argue this should wait until actual bus/DCA captures exist. The counterpoint is that bus balance, DCA, master-bus, mix-balance, and performance-adaptation promotion need a structured place for future physical evidence, and landing the intake as read-only does not widen operational authority.
