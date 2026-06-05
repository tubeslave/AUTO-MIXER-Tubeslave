# Codex Proposal: AutoFaderV2 Non-Fader Writes Fail Closed - 2026-06-05

## Short thesis

Keep AutoFaderV2 realtime analysis, but make its gate and EQ side effects fail closed so they emit blocked evidence instead of mutating the mixer.

## What I understood

The project needs an autonomous AI Mixing Engineer, but the current live-write proof boundary is still narrow. AutoFaderV2 has useful adaptive gate and masking EQ logic, yet those branches can call direct mixer methods outside the Product Layer fader/gain gate.

## Proposed solution

- Add a small non-fader write telemetry/result helper in `auto_fader_v2/controller.py`.
- Convert adaptive noise gate close/open decisions to blocked `set_gate_on` evidence.
- Convert the old fader-mute gate fallback to blocked evidence as well.
- Convert spectral masking and cross-adaptive EQ adjustments to blocked `set_eq_band` evidence.
- Expose last blocked non-fader telemetry in `get_status()`.
- Update write-surface inventory, capability matrix, processing inventory and focused tests.

## Likely files to touch

- `backend/auto_fader_v2/controller.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_auto_fader_v2_non_fader_fail_closed.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/auto-fader-v2-non-fader-writes-fail-closed-20260605.md`

## Alternatives considered

- Leave these writes because they are hidden inside AutoFaderV2. Rejected because direct controller use can bypass Product Layer safety.
- Remove adaptive gate and masking EQ logic. Rejected because the would-be decisions are useful evidence for future supervised promotion.
- Promote gate/EQ writes now. Rejected because false-open/false-close replay, filter audit, readback, rollback and hardware proof are incomplete.
- Migrate all fader writes in AutoFaderV2 at the same time. Rejected for this slice because fader apply is a separate Product Layer migration with a wider blast radius.

## Risks

- Callers that expected adaptive gate or masking EQ to mutate the mixer now receive blocked evidence. This is intentional.
- `_gate_closed_channels` now tracks desired blocked gate state rather than physical gate state for this quarantined branch.

## Test plan

- Add focused fake-mixer tests for adaptive gate close/open and masking EQ blocked evidence.
- Add a static test proving no direct `mixer_client.set_gate_on` or `mixer_client.set_eq_band` calls remain in AutoFaderV2.
- Run AutoFaderV2, registry, matrix and write-surface tests.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer first registering AutoFaderV2 as its own processing method. That can be useful later, but the immediate safety risk is hidden non-fader mutation, so this patch keeps metadata scoped to support paths and write-surface inventory without inventing a new approved method.
