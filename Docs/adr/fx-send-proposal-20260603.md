# fx-send-proposal-20260603

## 1. Context

The long-term project goal requires automatic effects management, effect send
tuning, and console FX control. Existing FX planning code includes useful rules
but also an apply path. The current live runtime must remain supervised
fader/gain only.

## 2. Options considered

- Promote `auto_fx.py` into the main assisted pipeline.
- Keep FX lab-only until supervised FX writes are proven.
- Add a separate transport-free FX send proposal layer.

## 3. Decision

Add `backend/fx_send_proposal.py` as a proposal-only analyzer. It estimates FX
send and return settings from source role, tempo, current send, arrangement
role, contribution, return headroom, feedback risk, intelligibility risk, and
persistence. It does not create `requested_change` payloads for live apply.

## 4. Why this won

This moves effects and sends from lab/quarantined metadata into a usable Product
Layer proposal surface while preserving the current live-write boundary. It
separates FX reasoning from legacy transport/apply code.

## 5. Rejected alternatives

Direct reuse of `auto_fx.py` risks mixing proposal logic with console writes.
Keeping FX lab-only leaves explicit project-goal capabilities without a
product-visible proposal surface.

## 6. Implementation plan

1. Build bounded FX send analysis and recommendation objects.
2. Allow source-role-backed send proposals for vocals, drums, guitars, keys,
   percussion, and playback.
3. Block dry-anchor, feedback-risk, intelligibility-risk, and low-headroom
   cases.
4. Emit proposal-only payloads with no live apply fields.
5. Register the method and proposal contract.
6. Update the project-goal capability matrix to assisted/proposal-only.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_fx_send_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness. The method is registered as
proposal-only, emits no queueable live changes, and includes blockers for
unsupported FX value type and missing hardware proof. Rollback is removal of
the module, registry entry, proposal contract, matrix update, tests, brief,
proposal, and ADR; no live transport code is touched.
