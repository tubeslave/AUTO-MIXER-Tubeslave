# Automixer WING Read-Only Preflight Backend-Connected Attempt - 2026-06-09 (Timeout)

## What was done

- Started the local backend server from `backend/server.py`; it reported:
  - `AutoMixer Server initialized on 0.0.0.0:8765`
  - `Starting WebSocket server on 0.0.0.0:8765`
- Ran the WING read-only preflight offline-plan + real run flow from
  `Docs/runbooks/WING_READONLY_PREFLIGHT.md`:
  - `--plan-only` generated a plan at:
    - `/tmp/wing_preflight_plan2/latest_plan.json`
  - Real run used `--expected-plan` and executed against `ws://localhost:8765`

## Result

- `status`: `blocked`
- `run_id`: `20260609T115911Z_wing_readonly_preflight`
- `reason`: `read_only_preflight_timeout`
- `blockers`: `["transport_timeout_or_connection_loss"]`
- `expected_plan_match`: `true`
- `expected_plan_required_for_real_run`: `true`
- `proof_artifact_schema`: `wing_readonly_preflight.v1`
- `proof_runner`: `wing_readonly_preflight_runner`
- `read_only`: `true`
- `live_write`: `false`
- `write_request_sent`: `false`
- `proposal_request_sent`: `false`
- `forbidden_request_types_sent`: `[]`
- `transport_failure.type`: `TimeoutError`
- Artifact folder:
  - `/tmp/wing_preflight_run_20260609/20260609T115911Z_wing_readonly_preflight/`
    - `summary.json`
    - `events.jsonl`

## Interpretation

This proves the current blocker is not TCP/WebSocket reachability anymore; the backend is
reachable and starts, but it does not deliver the expected WING read-only preflight readback
sequence (transport timeout while awaiting query responses). The preflight remains fail-closed.

## Required next action

Continue with `Docs/runbooks/WING_READONLY_PREFLIGHT.md` when the intended backend is connected
to an actual WING Rack session and can return readback responses, then review and persist a
clean candidate before supervised proof steps.
