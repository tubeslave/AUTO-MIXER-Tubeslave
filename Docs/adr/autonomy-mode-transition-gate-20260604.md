# ADR: Autonomy Mode Transition Gate

Date: 2026-06-04

## 1. Context

The target architecture has four modes: Analysis, Assisted Mixing, Supervised
AutoMix, and Full Autonomous Mixing. The backend already exposes operator mode
policy, while readiness/proof layers explain safety and evidence. Those views
need a read-only gate that connects mode availability to current proof state.

## 2. Options considered

- Keep mode availability inferred in the UI.
- Add mode gating directly to runtime operator mode policy.
- Add a separate read-only transition gate.

## 3. Decision

Add `autonomy_mode_transition_gate`. It reports per-mode status, runtime entry
availability, current live apply scope, blockers, and next required actions.

## 4. Why this won

It makes the four-mode architecture auditable without changing runtime
behavior. Analysis and Assisted remain available. Supervised can be shown as
limited to the existing narrow fader/gain policy. Full Autonomous remains
blocked until proof gates cover the broader target.

## 5. Rejected alternatives

UI-only inference was rejected because mode gates should be testable. Runtime
policy mutation was rejected because current proof reports should not directly
change backend mode. Treating remediation as a reason to block the existing
narrow supervised mode was rejected because current policy already supports
explicitly supervised fader/gain apply.

## 6. Implementation plan

1. Add a pure mode gate builder.
2. Combine operator policy, static promotion readiness, and proof overlay.
3. Embed the gate in supervised proof report output.
4. Register it as shadow-only in the processing registry.
5. Add focused tests for mode availability, narrow supervised scope, and auto
   blocking.

## 7. Test plan

- Compile touched Python files.
- Run focused gate/report/registry tests.
- Run broader mode/proof/readiness/no-WING tests.
- Run the standard full pytest command.

## 8. Risks and rollback

Risk: limited supervised availability is mistaken for authority expansion. The
gate reports `live_widening_allowed=false`, `automatic_mode_change_allowed=false`,
and keeps `autonomous_apply_allowed_modes` empty. Rollback is to remove report
embedding and registry registration; no live path is introduced.
