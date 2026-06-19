# Plan M015: Replay threshold policy binding

## Checkpoints

| Checkpoint | Scope | Files | Evidence requirement | Safety boundary | Validation | Exit condition | Rollback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M015-CP01 | Bind offline replay promotion to the quality threshold policy contract | `backend/offline_replay_evaluation_contract.py`, `tests/test_offline_replay_evaluation_contract.py`, `tests/test_evaluation_integration_bundle_contract.py` | Replay manifests with only legacy threshold status fields are blocked; complete M014 threshold policy manifests are required | Offline replay evidence only, no hardware, no OSC writes | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py tests/test_evaluation_integration_bundle_contract.py -q` | replay result keeps `live_authority_granted: false` and `readiness_closure_allowed: false` | revert M015 commit | completed |
| M015-CP02 | Bind M015 to roadmap/current-state rails | `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`, `Docs/PROJECT_COMPLETION_GATE.md`, `Docs/adr/replay-threshold-policy-binding-20260619.md`, `tests/test_wing_mvp_rails.py` | M015 remains pending and proves M011 cannot bypass M014 with loose approval fields | Report-only gate binding | `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m015_replay_threshold_policy_binding_requires_m014_manifest -q` | roadmap/capability/current-state/report rows | revert M015 commit | completed |

## Validation Contract

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_evaluation_integration_bundle_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m015_replay_threshold_policy_binding_requires_m014_manifest -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Acceptance Criteria

- `offline_replay_evaluation_contract` evaluates `quality_threshold_policy_manifest`.
- Legacy `threshold_policy` fields without the M014 contract manifest do not
  satisfy approved-threshold requirements.
- M014 blockers are surfaced with the `quality_threshold_policy:` prefix.
- Integration bundles surface replay threshold-policy blockers through the
  existing `offline_replay:` child prefix.
- Replay threshold policy binding cannot grant live authority, close readiness,
  contact WING/backend, send live writes or become hardware proof.
